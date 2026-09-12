"""Q4 V3：不知道源数时仍能严格结束的搜索、定位与清除控制器。"""

from __future__ import annotations

import math
import time

from q4_channel_ledger import ChannelLedger
from q4_controller_v2 import Q4V2Controller
from q4_localization import bearing_intersection_score, ready_to_clear
from q4_models import ChannelRecord, ChannelStatus, Point, Q4Config, TriangleMesh
from q4_task_scheduler_v2 import RouteTask
from q4_task_scheduler_v3 import certificate_priority_order, choose_global_task


class Q4V3Controller(Q4V2Controller):
    """正式口径控制器；不会读取或接受本场干扰源数量。"""

    def __init__(self, client, config: Q4Config):
        super().__init__(client, config, expected_sources=None, expected_source_provenance="forbidden")
        self.opportunistic_attempts = {channel: 0 for channel in range(1, config.channel_count + 1)}
        self.discovery_vertex = {channel: None for channel in range(1, config.channel_count + 1)}
        self.metrics.update({
            "runtime_source_count_used": False,
            "opportunistic_positive": 0,
            "global_task_steps": 0,
        })

    def _resolve_expected_sources(self, enter_response: dict) -> None:
        """有意忽略所有源数相关字段，防止离线元数据泄露进算法。"""
        self.expected_sources = None
        self.source_count_provenance = "unknown_by_problem_statement"

    def _worth_free_route_measurement(self, record: ChannelRecord, point: Point) -> bool:
        if record.status != ChannelStatus.DETECTED or ready_to_clear(record, self.config):
            return False
        if self.opportunistic_attempts[record.channel] >= 5 or len(record.observations) >= 3:
            return False
        if any(math.dist(point, old) < self.config.min_localization_baseline_m for old in record.measured_positions):
            return False
        if record.circle_center is None:
            return False
        if math.dist(point, record.circle_center) > self.config.maximum_radius_m + record.circle_radius_m:
            return False
        return bearing_intersection_score(point, record) >= 0.16

    def _scan(self, mesh: TriangleMesh) -> int:
        self.route = certificate_priority_order(self.position, mesh, self.config)
        for sequence, vertex_id in enumerate(self.route):
            point = mesh.vertices[vertex_id]
            unknown = self._unknown_order(sequence)
            for record in unknown:
                before = record.status
                self._measure(point, record, "search", vertex_id)
                if before == ChannelStatus.UNKNOWN and record.status == ChannelStatus.DETECTED:
                    self.discovery_vertex[record.channel] = sequence + 1

            candidates = [
                record for record in self.records.values()
                if self._worth_free_route_measurement(record, point)
            ]
            candidates.sort(key=lambda record: (-bearing_intersection_score(point, record), record.channel))
            for record in candidates:
                self.opportunistic_attempts[record.channel] += 1
                result = self._measure(point, record, "opportunistic", vertex_id)
                if result in {"near", "direction"}:
                    self.metrics["opportunistic_positive"] += 1

            ready_here = [
                record for record in self.records.values()
                if record.status == ChannelStatus.DETECTED
                and ready_to_clear(record, self.config)
                and record.circle_center is not None
                and math.dist(self.position, record.circle_center) <= 100.0
            ]
            for record in sorted(ready_here, key=lambda item: math.dist(self.position, item.circle_center)):
                assert record.circle_center is not None
                self._clear(record.circle_center, record)
        return len(self.route)

    def _global_localize_and_clear(self, mesh: TriangleMesh) -> None:
        attempts = {channel: 0 for channel in self.records}
        while True:
            detected = [record for record in self.records.values() if record.status == ChannelStatus.DETECTED]
            if not detected:
                return
            tasks: list[RouteTask] = []
            for record in detected:
                if attempts[record.channel] > 2 * self.config.max_localization_measurements:
                    raise RuntimeError(f"频道{record.channel}达到定位补测上限")
                tasks.append(self._localization_task(record, mesh))
            task = choose_global_task(self.position, tasks)
            record = self.records[task.channel]
            attempts[record.channel] += 1
            self.metrics["global_task_steps"] += 1
            if task.kind == "clear":
                self._clear(task.position, record)
            elif task.kind == "try_clear":
                self.speculative_clear_points[record.channel].append(task.position)
                self.metrics["speculative_clear_attempts"] += 1
                self._clear(task.position, record, certified=False)
            else:
                self._measure(task.position, record, "localization", task.vertex_id)

    def run(self, mesh: TriangleMesh) -> dict:
        started = time.perf_counter()
        self.ledger = ChannelLedger(self.config.channel_count, mesh)
        enter_response = self.client.enter()
        self._resolve_expected_sources(enter_response)
        visited = self._scan(mesh)
        self._mark_absent_after_full_scan(mesh, visited)
        self._global_localize_and_clear(mesh)
        self._mark_absent_after_full_scan(mesh, visited)

        cleared = sum(record.status == ChannelStatus.CLEARED for record in self.records.values())
        absent = sum(record.status == ChannelStatus.ABSENT for record in self.records.values())
        unresolved = [
            record.channel for record in self.records.values()
            if record.status not in {ChannelStatus.CLEARED, ChannelStatus.ABSENT}
        ]
        if unresolved or cleared + absent != self.config.channel_count or not 10 <= cleared <= 16:
            raise RuntimeError(f"未知源数严格停止失败：清除{cleared}，不存在{absent}，未解决{unresolved}")

        exit_response = self.client.exit()
        virtual_time = float(exit_response["virtual_time_s"])
        discovered = [value for value in self.discovery_vertex.values() if value is not None]
        self.metrics.update({
            "source_count_provenance": self.source_count_provenance,
            "runtime_source_count_used": False,
            "triangle_vertices_visited": visited,
            "total_triangle_vertices": len(mesh.vertices),
            "cleared_count": cleared,
            "absent_count": absent,
            "strict_termination_verified": True,
            "last_discovery_vertex": max(discovered, default=None),
            "virtual_time_s": virtual_time,
            "average_time_per_cleared_s": virtual_time / cleared,
            "wall_time_s": time.perf_counter() - started,
            "scheduler_version": "Q4_V3_unknown_count_certificate_joint",
        })
        return dict(self.metrics)
