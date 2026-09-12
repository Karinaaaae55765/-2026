"""Q4 V5：25点严格证书、Q2补测和覆盖定位统一调度。"""

from __future__ import annotations

import math
import time

import numpy as np

from q4_channel_ledger import ChannelLedger
from q4_controller_v4 import Q4V4Controller
from q4_dynamic_certificate_v4 import certificate_status
from q4_localization import bearing_intersection_score, ready_to_clear
from q4_models import ChannelRecord, ChannelStatus, Point, Q4Config, TriangleMesh
from q4_state_filter_v5 import ConservativeDirectionalFilter, v5_certificate_template
from q4_task_scheduler_v3 import choose_global_task
from q4_task_scheduler_v5 import V5Task, choose_joint_task, optimized_open_order


class Q4V5Controller(Q4V4Controller):
    """低源数优先；运行时不读取真实源数，严格证书始终保留。"""

    def __init__(self, client, config: Q4Config):
        super().__init__(client, config)
        self.direction_filter = ConservativeDirectionalFilter(config)
        self.q2_attempts = {channel: 0 for channel in range(1, config.channel_count + 1)}
        self.short_wedge_attempted = {channel: False for channel in range(1, config.channel_count + 1)}
        self.metrics.update({
            "joint_scheduler_calls": 0,
            "q2_probe_measurements": 0,
            "q2_probe_positive": 0,
            "state_filter_absent": 0,
            "state_box_score_evaluations": 0,
            "state_box_best_score": 0,
            "certificate_template_points": len(v5_certificate_template()),
            "blind_wedge_used": False,
        })

    def _measure(self, point: Point, record: ChannelRecord, phase: str, vertex_id: int | None = None) -> str:
        result = super()._measure(point, record, phase, vertex_id)
        self.direction_filter.record(record.channel, point, result)
        return result

    def _q2_candidates(self, record: ChannelRecord) -> list[Point]:
        """围绕单示向楔形中部构造对称侧向候选，兼顾交会角与最坏距离。"""
        if len(record.observations) != 1 or record.circle_center is None:
            return []
        observation = record.observations[0]
        angle = math.radians(observation.bearing_deg)
        along = np.asarray((math.cos(angle), math.sin(angle)), dtype=float)
        lateral = np.asarray((-along[1], along[0]), dtype=float)
        center = np.asarray(record.circle_center, dtype=float)
        radius = record.circle_radius_m if math.isfinite(record.circle_radius_m) else 750.0
        lateral_step = max(160.0, min(360.0, self.config.minimum_radius_m - radius - 35.0))
        candidates: list[Point] = []
        # 锚点已经得到正响应。围绕锚点作侧向成对补测，对圆周朝外源尤其重要：
        # 不会像只在楔形中点补测那样越过源位置、整体落到发射背面。
        anchor = np.asarray(observation.position, dtype=float)
        for backward in (120.0, 0.0):
            for sign in (1.0, -1.0):
                value = anchor - backward * along + sign * 220.0 * lateral
                point = (float(value[0]), float(value[1]))
                if math.hypot(*point) <= self.config.arena_radius_m + 450.0:
                    candidates.append(point)
        for forward in (0.0, 180.0, -180.0):
            for sign in (1.0, -1.0):
                value = center + forward * along + sign * lateral_step * lateral
                point = (float(value[0]), float(value[1]))
                if math.hypot(*point) <= self.config.arena_radius_m + 100.0:
                    candidates.append(point)
        return [
            point for point in candidates
            if all(math.dist(point, old) >= self.config.min_localization_baseline_m for old in record.measured_positions)
        ]

    def _q2_task(self, record: ChannelRecord) -> V5Task | None:
        if self.q2_attempts[record.channel] >= 6:
            return None
        candidates = self._q2_candidates(record)
        if not candidates:
            return None
        point = max(candidates, key=lambda candidate: (
            bearing_intersection_score(candidate, record) / (1.0 + math.dist(self.position, candidate) / 700.0),
            -math.dist(candidate, record.observations[0].position),
            -math.dist(self.position, candidate),
        ))
        return V5Task("probe", point, record.channel, 350.0 * bearing_intersection_score(point, record), False)

    def _prefer_short_wedge(self, record: ChannelRecord) -> bool:
        """外环锚点向圆内示向时，目标通常紧邻边界，短楔形比远距离补测更省。"""
        if len(record.observations) != 1:
            return False
        observation = record.observations[0]
        angle = math.radians(observation.bearing_deg)
        inward_dot = observation.position[0] * math.cos(angle) + observation.position[1] * math.sin(angle)
        anchor_radius = math.hypot(*observation.position)
        radial_cosine = inward_dot / max(anchor_radius, 1.0)
        return anchor_radius >= self.config.arena_radius_m and radial_cosine <= -0.985

    def _execute_probe(self, task: V5Task) -> None:
        record = self.records[task.channel]
        result = self._measure(task.position, record, "localization")
        self.q2_attempts[record.channel] += 1
        self.metrics["q2_probe_measurements"] += 1
        if result in {"direction", "near"}:
            self.metrics["q2_probe_positive"] += 1

    def _try_short_boundary_wedge(self, record: ChannelRecord, maximum_distance: float = 320.0) -> bool:
        anchor = record.observations[0].position
        self.short_wedge_attempted[record.channel] = True
        for point in self._wedge_clear_path(record):
            if math.dist(anchor, point) > maximum_distance:
                break
            if any(math.dist(point, old) < 1.0 for old in self.speculative_clear_points[record.channel]):
                continue
            self.speculative_clear_points[record.channel].append(point)
            self.metrics["speculative_clear_attempts"] += 1
            self.metrics["wedge_clear_attempts"] += 1
            if self._clear(point, record, certified=False):
                self.metrics["wedge_clear_success"] += 1
                return True
        return False

    def _opportunistic_at(self, point: Point) -> None:
        candidates = [record for record in self.records.values() if self._worth_free_route_measurement(record, point)]
        candidates.sort(key=lambda record: (-bearing_intersection_score(point, record), record.channel))
        for record in candidates:
            self.opportunistic_attempts[record.channel] += 1
            result = self._measure(point, record, "opportunistic")
            if result in {"direction", "near"}:
                self.metrics["opportunistic_positive"] += 1

    def _joint_search(self, mesh: TriangleMesh) -> None:
        template = v5_certificate_template()
        pending = optimized_open_order(self.position, template)
        coverage_steps = 0
        while pending:
            if self._finish_by_cardinality():
                return
            status = certificate_status(self.dynamic_points, self.config.arena_radius_m, self.config.minimum_radius_m)
            if status.complete:
                for record in self._unknown_records():
                    record.status = ChannelStatus.ABSENT
                return

            next_coverage = pending[0]
            unknown_records = self._unknown_records()
            triangles = [tuple(mesh.vertices[index] for index in triangle) for triangle in mesh.triangles]
            state_scores: dict[Point, int] = {}
            if unknown_records:
                representative = unknown_records[0].channel
                for point in pending[:4]:
                    state_scores[point] = self.direction_filter.orientation_elimination_score(representative, point, triangles)
                self.metrics["state_box_score_evaluations"] += len(state_scores)
                self.metrics["state_box_best_score"] = max(self.metrics["state_box_best_score"], max(state_scores.values(), default=0))
            tasks = [
                V5Task(
                    "coverage",
                    point,
                    information=(450.0 if index == 0 else 260.0 / (1.0 + index)) + min(60.0, state_scores.get(point, 0) / 25.0),
                    certificate_point=True,
                )
                for index, point in enumerate(pending[:4])
            ]
            for record in self.records.values():
                if record.status != ChannelStatus.DETECTED:
                    continue
                if ready_to_clear(record, self.config) and record.circle_center is not None:
                    detour = math.dist(self.position, record.circle_center) + math.dist(record.circle_center, next_coverage) - math.dist(self.position, next_coverage)
                    if detour <= 650.0:
                        tasks.append(V5Task("clear", record.circle_center, record.channel))
                elif coverage_steps >= 3 and len(record.observations) == 1:
                    probe = self._q2_task(record)
                    if probe is not None:
                        detour = math.dist(self.position, probe.position) + math.dist(probe.position, next_coverage) - math.dist(self.position, next_coverage)
                        if detour <= 350.0:
                            tasks.append(probe)

            self.metrics["joint_scheduler_calls"] += 1
            task = choose_joint_task(self.position, tasks, len(self._unknown_records()), coverage_steps)
            if task.kind == "clear":
                self._clear(task.position, self.records[task.channel])
                continue
            if task.kind == "probe":
                self._execute_probe(task)
                coverage_steps = 0
                continue

            self._visit_shared_point(task.position, phase="coverage")
            pending = [point for point in pending if math.dist(point, task.position) >= 1.0]
            self._opportunistic_at(task.position)
            # 圆周朝外源若在外环点首次出现，必须当场处理；离开该扇区后再回来
            # 会产生数公里折返。这里的楔形从正响应锚点向内，通常仅百余米。
            boundary_single = [
                record for record in self.records.values()
                if record.status == ChannelStatus.DETECTED
                and self._prefer_short_wedge(record)
                and not self.short_wedge_attempted[record.channel]
                and math.dist(record.observations[0].position, task.position) < 1.0
            ]
            for record in boundary_single:
                self.metrics["blind_wedge_used"] = True
                self._try_short_boundary_wedge(record)
            coverage_steps += 1

        status = certificate_status(self.dynamic_points, self.config.arena_radius_m, self.config.minimum_radius_m)
        if not status.complete:
            self.dynamic_fallback_used = True
            for point in mesh.vertices:
                if self._finish_by_cardinality():
                    return
                if any(math.dist(point, old) < 1.0 for old in self.dynamic_points):
                    continue
                self._visit_shared_point(point, phase="coverage")
                self._opportunistic_at(point)
                if certificate_status(self.dynamic_points, self.config.arena_radius_m, self.config.minimum_radius_m).complete:
                    break
        status = certificate_status(self.dynamic_points, self.config.arena_radius_m, self.config.minimum_radius_m)
        if not status.complete:
            raise RuntimeError("V5动态证书与后备网格均未完成")
        for record in self._unknown_records():
            record.status = ChannelStatus.ABSENT

    def _finish_localization(self, mesh: TriangleMesh) -> None:
        attempts = {channel: 0 for channel in self.records}
        while True:
            detected = [record for record in self.records.values() if record.status == ChannelStatus.DETECTED]
            if not detected:
                return
            ready = [record for record in detected if ready_to_clear(record, self.config) and record.circle_center is not None]
            if ready:
                record = min(ready, key=lambda item: math.dist(self.position, item.circle_center))
                self._clear(record.circle_center, record)
                continue

            q2_tasks = [task for record in detected if len(record.observations) == 1 if (task := self._q2_task(record)) is not None]
            if q2_tasks:
                task = min(q2_tasks, key=lambda value: math.dist(self.position, value.position) - value.information)
                self._execute_probe(task)
                continue

            single = [record for record in detected if len(record.observations) == 1]
            if single:
                preferred = [record for record in single if self._prefer_short_wedge(record)]
                record = min(preferred or single, key=lambda item: math.dist(self.position, item.observations[0].position))
                self.metrics["blind_wedge_used"] = True
                if self._clear_single_bearing_wedge(record):
                    continue
                raise RuntimeError(f"频道{record.channel}严格楔形后备未找到目标")

            route_tasks = []
            for record in detected:
                attempts[record.channel] += 1
                if attempts[record.channel] > 2 * self.config.max_localization_measurements:
                    raise RuntimeError(f"频道{record.channel}达到V5定位补测上限")
                route_tasks.append(self._localization_task(record, mesh))
            task = choose_global_task(self.position, route_tasks)
            record = self.records[task.channel]
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
        self._resolve_expected_sources(self.client.enter())
        self._joint_search(mesh)
        self._finish_localization(mesh)
        self._finish_by_cardinality()

        cleared = sum(record.status == ChannelStatus.CLEARED for record in self.records.values())
        absent = sum(record.status == ChannelStatus.ABSENT for record in self.records.values())
        unresolved = [record.channel for record in self.records.values() if record.status not in {ChannelStatus.CLEARED, ChannelStatus.ABSENT}]
        if unresolved or cleared + absent != self.config.channel_count or not 10 <= cleared <= 16:
            raise RuntimeError(f"V5严格停止失败：清除{cleared}，不存在{absent}，未解决{unresolved}")
        response = self.client.exit()
        virtual_time = float(response["virtual_time_s"])
        status = certificate_status(self.dynamic_points, self.config.arena_radius_m, self.config.minimum_radius_m)
        self.metrics.update({
            "runtime_source_count_used": False,
            "source_count_provenance": "unknown_by_problem_statement",
            "dynamic_points_visited": len(self.dynamic_points),
            "dynamic_certificate_complete": status.complete,
            "dynamic_maximum_edge_m": status.maximum_intersecting_edge_m,
            "dynamic_fallback_used": self.dynamic_fallback_used,
            "absent_by_cardinality": self.absent_by_cardinality,
            "cleared_count": cleared,
            "absent_count": absent,
            "strict_termination_verified": True,
            "virtual_time_s": virtual_time,
            "average_time_per_cleared_s": virtual_time / cleared,
            "wall_time_s": time.perf_counter() - started,
            "scheduler_version": "Q4_V5_true_joint_low_count",
        })
        return dict(self.metrics)
