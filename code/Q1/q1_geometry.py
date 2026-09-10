"""Q1 deterministic bearing-intersection geometry.

Each bearing observation defines a closed angular wedge with half-width
``delta_deg``.  The feasible localization region is the intersection of all
wedge half-planes.  A bounded result is reduced to its convex hull, whose
diameter is computed with rotating calipers and checked against an O(m^2)
baseline.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

Point = tuple[float, float]
Vector = tuple[float, float]


@dataclass(frozen=True)
class Observation:
    x: float
    y: float
    bearing_deg: float

    @property
    def point(self) -> Point:
        return (self.x, self.y)


@dataclass(frozen=True)
class HalfPlane:
    """Half-plane represented by normal dot p >= offset."""

    normal: Vector
    offset: float


@dataclass(frozen=True)
class Boundary:
    point: Point
    direction: Vector


def cross(a: Vector, b: Vector) -> float:
    return a[0] * b[1] - a[1] * b[0]


def subtract(a: Point, b: Point) -> Vector:
    return (a[0] - b[0], a[1] - b[1])


def squared_distance(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def normalize_bearing(angle_deg: float) -> float:
    return angle_deg % 360.0


def unit_vector(angle_deg: float) -> Vector:
    theta = math.radians(normalize_bearing(angle_deg))
    return (math.cos(theta), math.sin(theta))


def _coerce_observations(items: Sequence[Observation | Mapping[str, Any]]) -> list[Observation]:
    if not items:
        raise ValueError("observations must contain at least one record")
    observations: list[Observation] = []
    for index, item in enumerate(items):
        if isinstance(item, Observation):
            obs = item
        elif isinstance(item, Mapping):
            missing = {"x", "y", "bearing_deg"} - set(item)
            if missing:
                raise ValueError(f"observation {index} missing fields: {sorted(missing)}")
            obs = Observation(float(item["x"]), float(item["y"]), float(item["bearing_deg"]))
        else:
            raise TypeError(f"observation {index} must be an Observation or mapping")
        if not all(math.isfinite(value) for value in (obs.x, obs.y, obs.bearing_deg)):
            raise ValueError(f"observation {index} contains a non-finite value")
        observations.append(obs)
    return observations


def build_geometry(
    observations: Sequence[Observation], delta_deg: float
) -> tuple[list[HalfPlane], list[Boundary]]:
    halfplanes: list[HalfPlane] = []
    boundaries: list[Boundary] = []
    for obs in observations:
        lower = unit_vector(obs.bearing_deg - delta_deg)
        upper = unit_vector(obs.bearing_deg + delta_deg)

        # cross(lower, p-s) >= 0
        normal_lower = (-lower[1], lower[0])
        halfplanes.append(
            HalfPlane(normal_lower, normal_lower[0] * obs.x + normal_lower[1] * obs.y)
        )
        boundaries.append(Boundary(obs.point, lower))

        # cross(p-s, upper) >= 0
        normal_upper = (upper[1], -upper[0])
        halfplanes.append(
            HalfPlane(normal_upper, normal_upper[0] * obs.x + normal_upper[1] * obs.y)
        )
        boundaries.append(Boundary(obs.point, upper))
    return halfplanes, boundaries


def intersect_boundaries(a: Boundary, b: Boundary, tol_parallel: float) -> Point | None:
    denominator = cross(a.direction, b.direction)
    if abs(denominator) <= tol_parallel:
        return None
    delta = subtract(b.point, a.point)
    parameter = cross(delta, b.direction) / denominator
    point = (
        a.point[0] + parameter * a.direction[0],
        a.point[1] + parameter * a.direction[1],
    )
    return point if all(math.isfinite(value) for value in point) else None


def _constraint_residual(halfplane: HalfPlane, point: Point) -> float:
    return (
        halfplane.normal[0] * point[0]
        + halfplane.normal[1] * point[1]
        - halfplane.offset
    )


def point_is_feasible(point: Point, halfplanes: Sequence[HalfPlane], tol: float) -> bool:
    return all(_constraint_residual(halfplane, point) >= -tol for halfplane in halfplanes)


def point_in_all_wedges(
    point: Point,
    observations: Sequence[Observation | Mapping[str, Any]],
    delta_deg: float = 1.0,
    tol: float = 1e-9,
) -> bool:
    coerced = _coerce_observations(observations)
    halfplanes, _ = build_geometry(coerced, delta_deg)
    return point_is_feasible(point, halfplanes, tol)


def has_recession_direction(
    halfplanes: Sequence[HalfPlane], boundaries: Sequence[Boundary], tol_direction: float = 1e-12
) -> bool:
    """Return whether the homogeneous half-planes admit a nonzero direction."""
    candidates: list[Vector] = []
    for boundary in boundaries:
        candidates.append(boundary.direction)
        candidates.append((-boundary.direction[0], -boundary.direction[1]))
    for direction in candidates:
        if all(
            halfplane.normal[0] * direction[0] + halfplane.normal[1] * direction[1]
            >= -tol_direction
            for halfplane in halfplanes
        ):
            return True
    return False


def unique_points(points: Iterable[Point], tol: float) -> list[Point]:
    unique: list[Point] = []
    threshold_sq = tol * tol
    for point in sorted(points):
        if all(squared_distance(point, other) > threshold_sq for other in unique):
            unique.append(point)
    return unique


def convex_hull(points: Sequence[Point], tol_area: float = 0.0) -> list[Point]:
    """Andrew monotone chain; collinear interior points are removed."""
    ordered = sorted(set(points))
    if len(ordered) <= 1:
        return ordered

    def turn(o: Point, a: Point, b: Point) -> float:
        return cross(subtract(a, o), subtract(b, o))

    lower: list[Point] = []
    for point in ordered:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], point) <= tol_area:
            lower.pop()
        lower.append(point)

    upper: list[Point] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], point) <= tol_area:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def polygon_area(vertices: Sequence[Point]) -> float:
    return 0.5 * abs(
        sum(cross(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices)))
    ) if len(vertices) >= 3 else 0.0


def _canonical_pair(a: Point, b: Point) -> tuple[Point, Point]:
    return (a, b) if a <= b else (b, a)


def diameter_bruteforce(vertices: Sequence[Point]) -> tuple[float, tuple[Point, Point] | None]:
    if not vertices:
        return math.nan, None
    if len(vertices) == 1:
        return 0.0, (vertices[0], vertices[0])
    best_sq = -1.0
    best_pair: tuple[Point, Point] | None = None
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            candidate_sq = squared_distance(vertices[i], vertices[j])
            pair = _canonical_pair(vertices[i], vertices[j])
            if candidate_sq > best_sq or (candidate_sq == best_sq and pair < best_pair):
                best_sq = candidate_sq
                best_pair = pair
    return math.sqrt(best_sq), best_pair


def diameter_rotating_calipers(
    vertices_ccw: Sequence[Point], tol_area: float = 0.0
) -> tuple[float, tuple[Point, Point] | None]:
    n = len(vertices_ccw)
    if n <= 2:
        return diameter_bruteforce(vertices_ccw)

    def doubled_area(i: int, ni: int, j: int) -> float:
        return abs(
            cross(
                subtract(vertices_ccw[ni], vertices_ccw[i]),
                subtract(vertices_ccw[j], vertices_ccw[i]),
            )
        )

    best_sq = -1.0
    best_pair: tuple[Point, Point] | None = None

    def consider(i: int, j: int) -> None:
        nonlocal best_sq, best_pair
        pair = _canonical_pair(vertices_ccw[i], vertices_ccw[j])
        candidate_sq = squared_distance(*pair)
        if candidate_sq > best_sq or (candidate_sq == best_sq and pair < best_pair):
            best_sq = candidate_sq
            best_pair = pair

    j = 1
    for i in range(n):
        ni = (i + 1) % n
        steps = 0
        while (
            doubled_area(i, ni, (j + 1) % n) > doubled_area(i, ni, j) + tol_area
            and steps < n
        ):
            j = (j + 1) % n
            steps += 1
        consider(i, j)
        consider(ni, j)
        next_j = (j + 1) % n
        if abs(doubled_area(i, ni, next_j) - doubled_area(i, ni, j)) <= tol_area:
            consider(i, next_j)
            consider(ni, next_j)
    return math.sqrt(best_sq), best_pair


def diameter_circle_report(
    vertices: Sequence[Point], pair: tuple[Point, Point], tol: float
) -> dict[str, Any]:
    a, b = pair
    center = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    radius = math.dist(a, b) / 2.0
    max_distance = max((math.dist(vertex, center) for vertex in vertices), default=0.0)
    return {
        "center": [center[0], center[1]],
        "radius": radius,
        "covers_region": max_distance <= radius + tol,
        "max_vertex_distance": max_distance,
    }


def localize_source(
    observations: Sequence[Observation | Mapping[str, Any]],
    delta_deg: float = 1.0,
    tol: float | None = None,
    tol_parallel: float = 1e-12,
) -> dict[str, Any]:
    """Construct the feasible region and compute its finite diameter when defined."""
    coerced = _coerce_observations(observations)
    if not math.isfinite(delta_deg) or not 0.0 < delta_deg < 90.0:
        raise ValueError("delta_deg must be finite and satisfy 0 < delta_deg < 90")
    if tol is not None and (not math.isfinite(tol) or tol <= 0.0):
        raise ValueError("tol must be a positive finite number or None")
    if not math.isfinite(tol_parallel) or tol_parallel <= 0.0:
        raise ValueError("tol_parallel must be a positive finite number")

    halfplanes, boundaries = build_geometry(coerced, delta_deg)
    raw_candidates: list[Point] = []
    for i, first in enumerate(boundaries):
        for second in boundaries[i + 1 :]:
            point = intersect_boundaries(first, second, tol_parallel)
            if point is not None:
                raw_candidates.append(point)

    scale = max(
        1.0,
        *(abs(value) for obs in coerced for value in (obs.x, obs.y)),
        *(abs(value) for point in raw_candidates for value in point),
    )
    geometry_tol = tol if tol is not None else 1e-9 * scale
    feasible = [
        point for point in raw_candidates if point_is_feasible(point, halfplanes, geometry_tol)
    ]
    feasible = unique_points(feasible, geometry_tol)

    diagnostics: dict[str, Any] = {
        "num_observations": len(coerced),
        "num_boundary_lines": len(boundaries),
        "num_candidate_intersections": len(raw_candidates),
        "num_feasible_intersections": len(feasible),
        "coordinate_scale": scale,
        "tolerance": geometry_tol,
        "tol_parallel": tol_parallel,
    }

    if not feasible:
        return {
            "status": "EMPTY",
            "vertices_ccw": [],
            "diameter": None,
            "diameter_pair": None,
            "diameter_circle": None,
            "diagnostics": diagnostics,
        }

    area_tol = geometry_tol * scale
    hull = convex_hull(feasible, area_tol)
    diagnostics["hull_area"] = polygon_area(hull)

    if has_recession_direction(halfplanes, boundaries):
        return {
            "status": "UNBOUNDED",
            "vertices_ccw": [list(point) for point in hull],
            "diameter": None,
            "diameter_pair": None,
            "diameter_circle": None,
            "diagnostics": diagnostics,
        }

    diameter, pair = diameter_rotating_calipers(hull, area_tol)
    baseline, baseline_pair = diameter_bruteforce(hull)
    difference = abs(diameter - baseline)
    diagnostics["diameter_bruteforce"] = baseline
    diagnostics["diameter_absolute_difference"] = difference
    if difference > max(geometry_tol, 1e-10 * max(1.0, baseline)):
        raise ArithmeticError("rotating-calipers diameter disagrees with brute-force baseline")

    status = "DEGENERATE" if len(hull) <= 2 or diagnostics["hull_area"] <= area_tol else "OK"
    assert pair is not None and baseline_pair is not None
    circle = diameter_circle_report(hull, pair, geometry_tol)
    return {
        "status": status,
        "vertices_ccw": [list(point) for point in hull],
        "diameter": diameter,
        "diameter_pair": [list(pair[0]), list(pair[1])],
        "diameter_circle": circle,
        "diagnostics": diagnostics,
    }


def plot_localization(
    observations: Sequence[Observation | Mapping[str, Any]],
    result: Mapping[str, Any],
    output_path: str | Path,
    delta_deg: float = 1.0,
) -> None:
    """Save a diagnostic plot for a bounded localization result."""
    if result["status"] not in {"OK", "DEGENERATE"}:
        raise ValueError("plotting requires a bounded result")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    coerced = _coerce_observations(observations)
    vertices = [tuple(point) for point in result["vertices_ccw"]]
    coordinates = [value for obs in coerced for value in (obs.x, obs.y)] + [
        value for point in vertices for value in point
    ]
    span = max(100.0, max(coordinates) - min(coordinates))
    ray_length = 1.5 * span

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    for index, obs in enumerate(coerced, start=1):
        ax.scatter(obs.x, obs.y, marker="^", s=55, color="#1f77b4")
        ax.annotate(f"S{index}", (obs.x, obs.y), xytext=(5, 5), textcoords="offset points")
        for angle in (obs.bearing_deg - delta_deg, obs.bearing_deg + delta_deg):
            direction = unit_vector(angle)
            ax.plot(
                [obs.x, obs.x + ray_length * direction[0]],
                [obs.y, obs.y + ray_length * direction[1]],
                color="#7f7f7f",
                linewidth=0.9,
                linestyle="--",
            )

    if len(vertices) >= 3:
        ax.add_patch(Polygon(vertices, closed=True, facecolor="#ff9896", edgecolor="#d62728", alpha=0.45))
    elif len(vertices) == 2:
        ax.plot(*zip(*vertices), color="#d62728", linewidth=3)
    elif vertices:
        ax.scatter(*vertices[0], color="#d62728", s=60)

    pair = [tuple(point) for point in result["diameter_pair"]]
    ax.plot(*zip(*pair), color="#2ca02c", linewidth=2.2, label=f"Diameter = {result['diameter']:.3f} m")
    circle = result["diameter_circle"]
    ax.add_patch(
        Circle(
            tuple(circle["center"]),
            circle["radius"],
            fill=False,
            color="#9467bd",
            linewidth=1.6,
            label="Diameter-endpoint circle",
        )
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m, east)")
    ax.set_ylabel("y (m, north)")
    ax.set_title("Q1 bearing-intersection localization")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input JSON file")
    parser.add_argument("--output", type=Path, help="Optional output JSON path")
    parser.add_argument("--plot", type=Path, help="Optional diagnostic PNG path")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    observations = payload["observations"]
    delta_deg = float(payload.get("delta_deg", 1.0))
    tolerance = payload.get("tol")
    tolerance = None if tolerance is None else float(tolerance)
    result = localize_source(observations, delta_deg=delta_deg, tol=tolerance)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    if args.plot:
        plot_localization(observations, result, args.plot, delta_deg=delta_deg)


if __name__ == "__main__":
    main()

