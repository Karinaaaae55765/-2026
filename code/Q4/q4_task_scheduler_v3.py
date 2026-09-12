"""Q4 V3：兼顾完整路线长度与三角证书前缀完成度的调度器。"""

from __future__ import annotations

import math

from q4_models import Point, Q4Config, TriangleMesh
from q4_task_scheduler_v2 import RouteTask, progressive_mesh_order


def _prefix_certificate_score(order: list[int], mesh: TriangleMesh) -> float:
    """越早完成更多三角形，得分越高。"""
    completion = {vertex: [] for vertex in range(len(mesh.vertices))}
    for triangle_id, triangle in enumerate(mesh.triangles):
        for vertex in triangle:
            completion[vertex].append(triangle_id)
    visited: set[int] = set()
    completed: set[int] = set()
    score = 0.0
    for vertex in order:
        visited.add(vertex)
        for triangle_id in completion[vertex]:
            if triangle_id not in completed and all(v in visited for v in mesh.triangles[triangle_id]):
                completed.add(triangle_id)
        score += len(completed)
    return score


def _route_length(start: Point, order: list[int], mesh: TriangleMesh) -> float:
    total = 0.0
    current = start
    for vertex in order:
        total += math.dist(current, mesh.vertices[vertex])
        current = mesh.vertices[vertex]
    return total


def _greedy_certificate_order(start: Point, mesh: TriangleMesh, distance_weight: float) -> list[int]:
    incident = {vertex: [] for vertex in range(len(mesh.vertices))}
    for triangle_id, triangle in enumerate(mesh.triangles):
        for vertex in triangle:
            incident[vertex].append(triangle_id)
    remaining = set(range(len(mesh.vertices)))
    visited: set[int] = set()
    order: list[int] = []
    current = start
    while remaining:
        def key(vertex: int) -> tuple[float, float, int]:
            completed = sum(all(v == vertex or v in visited for v in mesh.triangles[t]) for t in incident[vertex])
            advanced = sum(sum(v in visited for v in mesh.triangles[t]) for t in incident[vertex])
            distance = math.dist(current, mesh.vertices[vertex])
            utility = 10.0 * completed + 0.8 * advanced + 0.15 * len(incident[vertex])
            return (utility - distance_weight * distance / mesh.side_m, -distance, -vertex)
        chosen = max(remaining, key=key)
        remaining.remove(chosen)
        visited.add(chosen)
        order.append(chosen)
        current = mesh.vertices[chosen]
    return order


def certificate_priority_order(start: Point, mesh: TriangleMesh, config: Q4Config) -> list[int]:
    """从多个确定性候选中选择兼顾全程与前缀证书的路线。"""
    candidates = [progressive_mesh_order(start, mesh, config)]
    candidates.extend(_greedy_certificate_order(start, mesh, weight) for weight in (1.5, 2.5, 4.0, 6.0))
    shortest = min(_route_length(start, route, mesh) for route in candidates)
    feasible = [route for route in candidates if _route_length(start, route, mesh) <= 1.12 * shortest]
    return max(
        feasible,
        key=lambda route: (
            _prefix_certificate_score(route, mesh) - 0.002 * _route_length(start, route, mesh),
            -_route_length(start, route, mesh),
        ),
    )


def choose_global_task(current: Point, tasks: list[RouteTask]) -> RouteTask:
    if not tasks:
        raise ValueError("全局任务列表为空")
    kind_bias = {"clear": -450.0, "try_clear": -220.0, "measure": 0.0}
    return min(tasks, key=lambda task: (math.dist(current, task.position) + kind_bias.get(task.kind, 0.0), task.channel))
