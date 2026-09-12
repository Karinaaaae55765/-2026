"""Q4 V5按真实虚拟时间统一比较覆盖、补测与清除任务。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from q4_models import Point


@dataclass(frozen=True)
class V5Task:
    kind: str
    position: Point
    channel: int | None = None
    information: float = 0.0
    certificate_point: bool = False


def task_virtual_cost(current: Point, task: V5Task, unknown_channels: int) -> float:
    cost = math.dist(current, task.position) / 5.0
    if task.kind == "clear":
        return cost + 5.0
    if task.certificate_point:
        cost += 6.0 * unknown_channels
    if task.channel is not None:
        cost += 6.0
    return cost


def choose_joint_task(current: Point, tasks: list[V5Task], unknown_channels: int, coverage_debt: int) -> V5Task:
    if not tasks:
        raise ValueError("V5联合任务池为空")

    def key(task: V5Task) -> tuple[float, float, int]:
        cost = task_virtual_cost(current, task, unknown_channels)
        if task.kind == "clear":
            value = 1100.0
        elif task.kind == "probe":
            value = 300.0 + task.information
        else:
            value = 150.0 + task.information + 45.0 * min(coverage_debt, 4)
        return value / max(cost, 1.0), -cost, -(task.channel or 0)

    return max(tasks, key=key)


def open_route_length(start: Point, points: list[Point]) -> float:
    total, current = 0.0, start
    for point in points:
        total += math.dist(current, point)
        current = point
    return total


def two_opt_open(start: Point, points: list[Point]) -> list[Point]:
    """固定起点、自由终点的确定性2-opt。"""
    if len(points) < 3:
        return list(points)
    route = list(points)
    best = open_route_length(start, route)
    improved = True
    while improved:
        improved = False
        for first in range(len(route) - 1):
            for last in range(first + 1, len(route)):
                candidate = route[:first] + list(reversed(route[first:last + 1])) + route[last + 1:]
                value = open_route_length(start, candidate)
                if value + 1e-7 < best:
                    route, best, improved = candidate, value, True
    return route


def optimized_open_order(start: Point, points: list[Point]) -> list[Point]:
    """多首点最近邻后作2-opt，避免单一路径初始化。"""
    if not points:
        return []
    candidates: list[list[Point]] = []
    for forced in sorted(range(len(points)), key=lambda index: math.dist(start, points[index]))[: min(8, len(points))]:
        remaining = set(range(len(points)))
        remaining.remove(forced)
        order = [points[forced]]
        current = points[forced]
        while remaining:
            chosen = min(remaining, key=lambda index: (math.dist(current, points[index]), index))
            remaining.remove(chosen)
            order.append(points[chosen])
            current = points[chosen]
        candidates.append(two_opt_open(start, order))
    return min(candidates, key=lambda route: open_route_length(start, route))
