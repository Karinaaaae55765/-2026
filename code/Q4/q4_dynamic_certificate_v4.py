"""纯 Python 动态三角剖分及圆域覆盖证书。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from q4_models import Point
from q4_triangular_certificate import triangle_intersects_disk


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def convex_hull(points: list[Point]) -> list[Point]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique
    lower: list[Point] = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 1e-9:
            lower.pop()
        lower.append(point)
    upper: list[Point] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 1e-9:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def hull_covers_disk(points: list[Point], radius: float) -> bool:
    hull = convex_hull(points)
    if len(hull) < 3:
        return False
    for a, b in zip(hull, hull[1:] + hull[:1]):
        edge = math.dist(a, b)
        if edge <= 1e-12 or _cross(a, b, (0.0, 0.0)) / edge < radius - 1e-7:
            return False
    return True


def _circumcircle_contains(a: Point, b: Point, c: Point, p: Point) -> bool:
    ax, ay = a[0] - p[0], a[1] - p[1]
    bx, by = b[0] - p[0], b[1] - p[1]
    cx, cy = c[0] - p[0], c[1] - p[1]
    determinant = (
        (ax * ax + ay * ay) * (bx * cy - by * cx)
        - (bx * bx + by * by) * (ax * cy - ay * cx)
        + (cx * cx + cy * cy) * (ax * by - ay * bx)
    )
    orientation = _cross(a, b, c)
    return determinant > 1e-7 if orientation > 0 else determinant < -1e-7


def delaunay(points: list[Point]) -> list[tuple[int, int, int]]:
    """Bowyer-Watson；点数很小，优先可移植性与确定性。"""
    unique = list(dict.fromkeys((float(x), float(y)) for x, y in points))
    if len(unique) < 3:
        return []
    span = max(max(abs(x), abs(y)) for x, y in unique) + 1.0
    super_points = [(-20.0 * span, -10.0 * span), (20.0 * span, -10.0 * span), (0.0, 20.0 * span)]
    work = unique + super_points
    n = len(unique)
    triangles: list[tuple[int, int, int]] = [(n, n + 1, n + 2)]
    for point_id in range(n):
        bad = [triangle for triangle in triangles if _circumcircle_contains(*(work[i] for i in triangle), work[point_id])]
        edge_count: dict[tuple[int, int], int] = {}
        for triangle in bad:
            for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                edge = tuple(sorted((first, second)))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        triangles = [triangle for triangle in triangles if triangle not in bad]
        for edge, count in edge_count.items():
            if count == 1:
                triangle = (edge[0], edge[1], point_id)
                if abs(_cross(*(work[i] for i in triangle))) > 1e-8:
                    triangles.append(triangle)
    return [triangle for triangle in triangles if all(vertex < n for vertex in triangle)]


@dataclass(frozen=True)
class DynamicCertificateStatus:
    complete: bool
    hull_covers_arena: bool
    intersecting_triangles: int
    violating_triangles: tuple[tuple[int, int, int], ...]
    maximum_intersecting_edge_m: float


def certificate_status(points: list[Point], arena_radius: float, maximum_edge: float) -> DynamicCertificateStatus:
    hull_ok = hull_covers_disk(points, arena_radius)
    triangles = delaunay(points)
    violating: list[tuple[int, int, int]] = []
    maximum = 0.0
    intersecting = 0
    for triangle in triangles:
        vertices = tuple(points[index] for index in triangle)
        if not triangle_intersects_disk(vertices, arena_radius):
            continue
        intersecting += 1
        longest = max(math.dist(vertices[i], vertices[j]) for i in range(3) for j in range(i + 1, 3))
        maximum = max(maximum, longest)
        if longest >= maximum_edge - 1e-7:
            violating.append(triangle)
    return DynamicCertificateStatus(hull_ok and intersecting > 0 and not violating, hull_ok, intersecting, tuple(violating), maximum)


def initial_certificate_candidates(arena_radius: float) -> list[Point]:
    # 18 点外环取 1873 m：除覆盖圆域外，还使圆周附近朝外发射的源
    # 通常能被相邻多个外环点接收，避免 12 点外环只得到一次示向。
    outer_count = 18
    outer_radius = 1873.0
    outer = [(outer_radius * math.cos(2.0 * math.pi * i / outer_count), outer_radius * math.sin(2.0 * math.pi * i / outer_count)) for i in range(outer_count)]
    inner_count, inner_radius = 9, 995.0
    inner = [(inner_radius * math.cos(2.0 * math.pi * i / inner_count), inner_radius * math.sin(2.0 * math.pi * i / inner_count)) for i in range(inner_count)]
    return outer + inner


def refinement_candidates(points: list[Point], status: DynamicCertificateStatus) -> list[Point]:
    suggestions: list[Point] = []
    for triangle in status.violating_triangles:
        vertices = [points[index] for index in triangle]
        first, second = max(((vertices[i], vertices[j]) for i in range(3) for j in range(i + 1, 3)), key=lambda pair: math.dist(*pair))
        midpoint = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
        if not any(math.dist(midpoint, old) < 1.0 for old in points + suggestions):
            suggestions.append(midpoint)
    return suggestions
