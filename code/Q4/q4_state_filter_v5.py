"""Q4 V5 未知频道的保守位置—方向状态评分与严格证书账本。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from q4_dynamic_certificate_v4 import certificate_status
from q4_models import Point, Q4Config


def v5_certificate_template() -> list[Point]:
    """返回经穷举验证的25点严格模板（含原点）。"""
    outer_count, outer_radius = 12, 1875.0
    inner_count, inner_radius = 12, 950.0
    outer = [
        (outer_radius * math.cos(2.0 * math.pi * index / outer_count),
         outer_radius * math.sin(2.0 * math.pi * index / outer_count))
        for index in range(outer_count)
    ]
    offset = math.pi / outer_count
    inner = [
        (inner_radius * math.cos(2.0 * math.pi * index / inner_count + offset),
         inner_radius * math.sin(2.0 * math.pi * index / inner_count + offset))
        for index in range(inner_count)
    ]
    return [(0.0, 0.0), *outer, *inner]


def _minimum_dot_over_interval(vector: Point, start: float, end: float) -> float:
    """角区间内单位方向与固定向量点积的严格最小值。"""
    radius = math.hypot(*vector)
    if radius <= 1e-12:
        return 0.0
    phase = math.atan2(vector[1], vector[0])
    candidates = [start, end]
    k0 = math.floor((start - phase - math.pi) / (2.0 * math.pi)) - 1
    for k in range(k0, k0 + 5):
        angle = phase + math.pi + 2.0 * math.pi * k
        if start <= angle <= end:
            candidates.append(angle)
    return min(radius * math.cos(angle - phase) for angle in candidates)


def robustly_illuminates_triangle(point: Point, triangle: tuple[Point, Point, Point], angle_start: float, angle_end: float, minimum_radius: float) -> bool:
    """该检测点是否对整个三角形和整个方向区间都必在接收半平面内。"""
    if max(math.dist(point, vertex) for vertex in triangle) >= minimum_radius - 1e-8:
        return False
    return min(
        _minimum_dot_over_interval((point[0] - vertex[0], point[1] - vertex[1]), angle_start, angle_end)
        for vertex in triangle
    ) >= -1e-9


@dataclass
class ConservativeDirectionalFilter:
    """保存无信号点并给任务调度提供保守信息量；严格退出仍由连续几何证书决定。"""

    config: Q4Config
    no_signal_points: dict[int, list[Point]] = field(default_factory=dict)
    angle_bins: int = 72

    def __post_init__(self) -> None:
        self.no_signal_points = {channel: [] for channel in range(1, self.config.channel_count + 1)}

    def record(self, channel: int, point: Point, result: str) -> None:
        if result != "no_signal":
            return
        if not any(math.dist(point, old) < 1.0 for old in self.no_signal_points[channel]):
            self.no_signal_points[channel].append(point)

    def strict_absent(self, channel: int) -> bool:
        status = certificate_status(
            self.no_signal_points[channel],
            self.config.arena_radius_m,
            self.config.minimum_radius_m,
        )
        return status.complete

    def orientation_elimination_score(self, channel: int, point: Point, triangles: list[tuple[Point, Point, Point]]) -> int:
        """估计一次无信号可严格消去的三角形×方向箱数量，仅用于排序。"""
        existing = self.no_signal_points[channel]
        if any(math.dist(point, old) < 1.0 for old in existing):
            return 0
        width = 2.0 * math.pi / self.angle_bins
        score = 0
        for triangle in triangles:
            for index in range(self.angle_bins):
                start, end = index * width, (index + 1) * width
                if robustly_illuminates_triangle(point, triangle, start, end, self.config.minimum_radius_m):
                    score += 1
        return score
