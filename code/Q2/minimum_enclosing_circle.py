"""Deterministic minimum enclosing circle for a finite 2-D point set."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    radius: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x, self.y)


def _contains(circle: Circle, point: tuple[float, float], tol: float = 1e-10) -> bool:
    scale = max(1.0, abs(circle.x), abs(circle.y), circle.radius, abs(point[0]), abs(point[1]))
    return math.hypot(point[0] - circle.x, point[1] - circle.y) <= circle.radius + tol * scale


def _diameter_circle(a: tuple[float, float], b: tuple[float, float]) -> Circle:
    x = (a[0] + b[0]) / 2.0
    y = (a[1] + b[1]) / 2.0
    return Circle(x, y, math.hypot(a[0] - b[0], a[1] - b[1]) / 2.0)


def _circumcircle(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> Circle | None:
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    coordinate_scale = max(1.0, *(abs(value) for value in (*a, *b, *c)))
    if abs(d) <= 1e-12 * coordinate_scale * coordinate_scale:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return Circle(ux, uy, math.hypot(ux - ax, uy - ay))


def minimum_enclosing_circle(points: Iterable[Iterable[float]], seed: int = 2026) -> Circle:
    """Return the exact MEC of the supplied finite points (randomized incremental)."""
    array = np.asarray(list(points), dtype=float)
    if array.ndim != 2 or (array.size and array.shape[1] != 2):
        raise ValueError("points must have shape (n, 2)")
    if len(array) == 0:
        raise ValueError("minimum enclosing circle is undefined for an empty set")
    if not np.isfinite(array).all():
        raise ValueError("points contain non-finite values")

    ordered = [tuple(map(float, row)) for row in array]
    random.Random(seed).shuffle(ordered)
    circle: Circle | None = None
    for i, p in enumerate(ordered):
        if circle is not None and _contains(circle, p):
            continue
        circle = Circle(p[0], p[1], 0.0)
        for j in range(i):
            q = ordered[j]
            if _contains(circle, q):
                continue
            circle = _diameter_circle(p, q)
            for k in range(j):
                r = ordered[k]
                if _contains(circle, r):
                    continue
                candidate = _circumcircle(p, q, r)
                if candidate is None:
                    pair_circles = [_diameter_circle(p, q), _diameter_circle(p, r), _diameter_circle(q, r)]
                    feasible = [c0 for c0 in pair_circles if all(_contains(c0, z) for z in (p, q, r))]
                    circle = min(feasible, key=lambda c0: c0.radius)
                else:
                    circle = candidate
    assert circle is not None
    return circle
