"""Q4 V4 动态覆盖、射线定位与清除任务联合评分。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from q4_models import Point


@dataclass(frozen=True)
class JointTask:
    kind: str
    position: Point
    channel: int | None = None
    value: float = 0.0


def choose_joint_task(current: Point, tasks: list[JointTask], unknown_channels: int) -> JointTask:
    if not tasks:
        raise ValueError("联合任务池为空")
    kind_value = {"clear": 900.0, "ray": 500.0, "localize": 380.0, "coverage": 280.0}
    def score(task: JointTask) -> tuple[float, float, int]:
        distance_time = math.dist(current, task.position) / 5.0
        scan_time = 6.0 * unknown_channels
        shared_scan_credit = min(420.0, 20.0 * unknown_channels) if task.kind != "clear" else 0.0
        utility = kind_value.get(task.kind, 0.0) + task.value + shared_scan_credit
        cost = max(1.0, distance_time + (scan_time if task.kind != "clear" else 5.0))
        return (utility / cost, -distance_time, -(task.channel or 0))
    return max(tasks, key=score)


def nearest_open_order(start: Point, points: list[Point]) -> list[int]:
    remaining = set(range(len(points)))
    order: list[int] = []
    current = start
    while remaining:
        chosen = min(remaining, key=lambda index: (math.dist(current, points[index]), index))
        remaining.remove(chosen)
        order.append(chosen)
        current = points[chosen]
    return order
