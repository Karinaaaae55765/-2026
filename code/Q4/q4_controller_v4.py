"""Q4 V4 动态证书、射线逼近和数量上界联合控制器。"""

from __future__ import annotations

import math
import time

from q4_channel_ledger import ChannelLedger
from q4_controller_v3 import Q4V3Controller
from q4_dynamic_certificate_v4 import certificate_status, initial_certificate_candidates, refinement_candidates
from q4_localization import bearing_intersection_score, ready_to_clear
from q4_models import ChannelRecord, ChannelStatus, Point, Q4Config, TriangleMesh
from q4_task_scheduler_v2 import RouteTask
from q4_task_scheduler_v3 import choose_global_task
from q4_task_scheduler_v4 import JointTask, nearest_open_order


class Q4V4Controller(Q4V3Controller):
    def __init__(self, client, config: Q4Config):
        super().__init__(client, config)
        self.dynamic_points: list[Point] = []
        self.ray_attempts = {channel: 0 for channel in range(1, config.channel_count + 1)}
        self.ray_steps = {channel: 400.0 for channel in range(1, config.channel_count + 1)}
        self.absent_by_cardinality = 0
        self.dynamic_fallback_used = False
        self.metrics.update({"dynamic_points_visited": 0, "ray_measurements": 0, "ray_positive": 0})
        self.metrics.update({"wedge_clear_attempts": 0, "wedge_clear_success": 0})

    def _present_count(self) -> int:
        return sum(record.status in {ChannelStatus.DETECTED, ChannelStatus.CLEARED} for record in self.records.values())

    def _unknown_records(self) -> list[ChannelRecord]:
        return [record for record in self.records.values() if record.status == ChannelStatus.UNKNOWN]

    def _measure_unknown_at(self, point: Point) -> None:
        sequence = len(self.dynamic_points)
        for record in sorted(self._unknown_records(), key=lambda item: item.channel, reverse=bool(sequence % 2)):
            before = record.status
            self._measure(point, record, "search", vertex_id=None)
            if before == ChannelStatus.UNKNOWN and record.status == ChannelStatus.DETECTED:
                self.discovery_vertex[record.channel] = sequence + 1

    def _visit_shared_point(self, point: Point, target_channel: int | None = None, phase: str = "coverage") -> str | None:
        # 外环证书点及圆域内的顺路定位点参与动态剖分；圆域外射线点
        # 不扩张证书凸包，避免制造新的长边三角形。
        contributes = phase == "coverage" or math.hypot(*point) <= self.config.arena_radius_m + 1e-7
        if contributes and not any(math.dist(point, old) < 1e-7 for old in self.dynamic_points):
            self.dynamic_points.append(point)
        self._measure_unknown_at(point)
        target_result = None
        if target_channel is not None:
            record = self.records[target_channel]
            if record.status == ChannelStatus.DETECTED and not any(math.dist(point, old) < 1.0 for old in record.measured_positions):
                result = self._measure(point, record, "localization")
                target_result = result
                if phase == "ray":
                    self.metrics["ray_measurements"] += 1
                    if result in {"direction", "near"}:
                        self.metrics["ray_positive"] += 1
        return target_result

    def _ray_task(self, record: ChannelRecord) -> JointTask | None:
        if not record.observations or self.ray_attempts[record.channel] >= 10:
            return None
        anchor = record.observations[-1]
        attempt = self.ray_attempts[record.channel]
        offsets = (0.0, 0.0, 1.5, -1.5, 3.0, -3.0, 5.0, -5.0, 8.0, -8.0)
        angle = math.radians(anchor.bearing_deg + offsets[attempt])
        distance = self.ray_steps[record.channel]
        point = (anchor.position[0] + distance * math.cos(angle), anchor.position[1] + distance * math.sin(angle))
        if any(math.dist(point, old) < 1.0 for old in record.measured_positions):
            self.ray_attempts[record.channel] += 1
            return self._ray_task(record)
        return JointTask("ray", point, record.channel, value=180.0 / (1.0 + attempt))

    def _localization_task(self, record: ChannelRecord, mesh: TriangleMesh) -> RouteTask:
        if record.circle_center is not None and record.circle_radius_m <= 60.0 and len(self.speculative_clear_points[record.channel]) < 3:
            if all(math.dist(record.circle_center, old) >= 8.0 for old in self.speculative_clear_points[record.channel]):
                return RouteTask("try_clear", record.channel, record.circle_center)
        return super()._localization_task(record, mesh)

    def _wedge_clear_path(self, record: ChannelRecord) -> list[Point]:
        """用 28 m 方格蛇形覆盖单示向的 ±1°、1500 m 窄楔形。"""
        observation = record.observations[0]
        spacing = 28.0
        maximum_x = self.config.maximum_radius_m
        layers = []
        x = self.config.near_radius_m
        while x < maximum_x:
            layers.append(x)
            x += spacing
        layers.append(maximum_x)
        angle = math.radians(observation.bearing_deg)
        along = (math.cos(angle), math.sin(angle))
        lateral = (-along[1], along[0])
        path: list[Point] = []
        for layer_id, distance in enumerate(layers):
            half_width = distance * math.tan(math.radians(self.config.bearing_error_deg))
            intervals = max(1, math.ceil(2.0 * half_width / spacing))
            offsets = [0.0] if intervals == 1 else [-half_width + 2.0 * half_width * i / intervals for i in range(intervals + 1)]
            if layer_id % 2:
                offsets.reverse()
            for offset in offsets:
                path.append((
                    observation.position[0] + distance * along[0] + offset * lateral[0],
                    observation.position[1] + distance * along[1] + offset * lateral[1],
                ))
        return path

    def _clear_single_bearing_wedge(self, record: ChannelRecord) -> bool:
        for point in self._wedge_clear_path(record):
            if any(math.dist(point, old) < 1.0 for old in self.speculative_clear_points[record.channel]):
                continue
            self.speculative_clear_points[record.channel].append(point)
            self.metrics["speculative_clear_attempts"] += 1
            self.metrics["wedge_clear_attempts"] += 1
            if self._clear(point, record, certified=False):
                self.metrics["wedge_clear_success"] += 1
                return True
        return False

    def _global_localize_and_clear(self, mesh: TriangleMesh) -> None:
        attempts = {channel: 0 for channel in self.records}
        while True:
            detected = [record for record in self.records.values() if record.status == ChannelStatus.DETECTED]
            if not detected:
                return
            ready = [record for record in detected if ready_to_clear(record, self.config) and record.circle_center is not None]
            if ready:
                record = min(ready, key=lambda item: math.dist(self.position, item.circle_center))
                assert record.circle_center is not None
                self._clear(record.circle_center, record)
                continue

            single = [record for record in detected if len(record.observations) == 1 and self.ray_attempts[record.channel] < 10]
            if single:
                record = min(single, key=lambda item: math.dist(self.position, item.observations[0].position))
                if self._clear_single_bearing_wedge(record):
                    continue
                self.ray_attempts[record.channel] = 10

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

    def _candidate_tasks(self, pending_coverage: list[Point], mesh: TriangleMesh) -> list[JointTask]:
        tasks: list[JointTask] = []
        for point in pending_coverage:
            tasks.append(JointTask("coverage", point, value=80.0))
        for record in self.records.values():
            if record.status != ChannelStatus.DETECTED:
                continue
            if ready_to_clear(record, self.config) and record.circle_center is not None:
                tasks.append(JointTask("clear", record.circle_center, record.channel, value=500.0))
                continue
            ray = self._ray_task(record)
            if ray is not None:
                tasks.append(ray)
            else:
                try:
                    fallback = self._localization_task(record, mesh)
                except RuntimeError:
                    continue
                tasks.append(JointTask("clear" if fallback.kind in {"clear", "try_clear"} else "localize", fallback.position, record.channel, value=150.0))
        return tasks

    def _finish_by_cardinality(self) -> bool:
        if self._present_count() < 16:
            return False
        for record in self._unknown_records():
            record.status = ChannelStatus.ABSENT
            self.absent_by_cardinality += 1
        return True

    def _dynamic_search_and_localize(self, mesh: TriangleMesh) -> None:
        pending = initial_certificate_candidates(self.config.arena_radius_m)
        self._visit_shared_point((0.0, 0.0))
        maximum_iterations = 80
        for _ in range(maximum_iterations):
            if self._finish_by_cardinality():
                break
            status = certificate_status(self.dynamic_points, self.config.arena_radius_m, self.config.minimum_radius_m)
            if status.complete:
                for record in self._unknown_records():
                    record.status = ChannelStatus.ABSENT
                break
            pending = [point for point in pending if not any(math.dist(point, old) < 1.0 for old in self.dynamic_points)]
            if not pending:
                pending.extend(refinement_candidates(self.dynamic_points, status))
            if not pending:
                self.dynamic_fallback_used = True
                pending.extend(point for point in mesh.vertices if not any(math.dist(point, old) < 1.0 for old in self.dynamic_points))
            if not pending:
                raise RuntimeError("动态证书候选点已耗尽")
            next_point = pending[nearest_open_order(self.position, pending)[0]]

            # 只执行真正顺路的清除，防止联合评分把证书路线撕裂。
            ready = [record for record in self.records.values() if record.status == ChannelStatus.DETECTED and ready_to_clear(record, self.config) and record.circle_center is not None]
            if ready:
                record = min(ready, key=lambda item: math.dist(self.position, item.circle_center))
                assert record.circle_center is not None
                detour = math.dist(self.position, record.circle_center) + math.dist(record.circle_center, next_point) - math.dist(self.position, next_point)
                if detour <= 1500.0:
                    self._clear(record.circle_center, record)

            self._visit_shared_point(next_point)
            pending.remove(next_point)

            # 在已经到达的证书点补测少量高交会角频道，不增加移动距离。
            candidates = [record for record in self.records.values() if self._worth_free_route_measurement(record, next_point)]
            candidates.sort(key=lambda record: (-bearing_intersection_score(next_point, record), record.channel))
            for record in candidates:
                self.opportunistic_attempts[record.channel] += 1
                result = self._measure(next_point, record, "opportunistic")
                if result in {"direction", "near"}:
                    self.metrics["opportunistic_positive"] += 1
        else:
            raise RuntimeError("动态联合调度达到迭代上限")

    def run(self, mesh: TriangleMesh) -> dict:
        started = time.perf_counter()
        self.ledger = ChannelLedger(self.config.channel_count, mesh)
        self._resolve_expected_sources(self.client.enter())
        self._dynamic_search_and_localize(mesh)
        self._global_localize_and_clear(mesh)
        self._finish_by_cardinality()

        cleared = sum(record.status == ChannelStatus.CLEARED for record in self.records.values())
        absent = sum(record.status == ChannelStatus.ABSENT for record in self.records.values())
        unresolved = [record.channel for record in self.records.values() if record.status not in {ChannelStatus.CLEARED, ChannelStatus.ABSENT}]
        if unresolved or cleared + absent != self.config.channel_count or not 10 <= cleared <= 16:
            raise RuntimeError(f"V4严格停止失败：清除{cleared}，不存在{absent}，未解决{unresolved}")
        exit_response = self.client.exit()
        virtual_time = float(exit_response["virtual_time_s"])
        status = certificate_status(self.dynamic_points, self.config.arena_radius_m, self.config.minimum_radius_m)
        self.metrics.update({
            "runtime_source_count_used": False,
            "source_count_provenance": "unknown_by_problem_statement",
            "dynamic_points_visited": len(self.dynamic_points),
            "dynamic_certificate_complete": status.complete,
            "dynamic_maximum_edge_m": status.maximum_intersecting_edge_m,
            "absent_by_cardinality": self.absent_by_cardinality,
            "dynamic_fallback_used": self.dynamic_fallback_used,
            "cleared_count": cleared,
            "absent_count": absent,
            "strict_termination_verified": True,
            "virtual_time_s": virtual_time,
            "average_time_per_cleared_s": virtual_time / cleared,
            "wall_time_s": time.perf_counter() - started,
            "scheduler_version": "Q4_V4_dynamic_certificate_ray_joint",
        })
        return dict(self.metrics)
