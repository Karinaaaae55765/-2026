"""Q2鲁棒第二检测点优化器：P2 = Omega1 与 W2 的交集。"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

Q1_DIR = Path(__file__).resolve().parents[1] / "Q1"
if str(Q1_DIR) not in sys.path:
    sys.path.insert(0, str(Q1_DIR))
from q1_geometry import (  # noqa: E402
    HalfPlane,
    Observation,
    build_geometry,
    diameter_rotating_calipers,
    localize_source,
)

from minimum_enclosing_circle import Circle, minimum_enclosing_circle


@dataclass(frozen=True)
class Q2Config:
    s1_x: float = 0.0
    s1_y: float = 0.0
    first_bearing_deg: float = 0.0
    bearing_error_deg: float = 1.0
    arena_x: float = 0.0
    arena_y: float = 0.0
    arena_radius_m: float = 1800.0
    near_radius_m: float = 5.0
    guaranteed_radius_m: float = 1000.0
    maximum_radius_m: float = 1500.0
    disk_sides: int = 128
    scenario_pool_count: int = 900
    candidate_step_m: float = 100.0
    refinement_step_m: float = 25.0
    scenario_count: int = 41
    error_scenario_count: int = 7
    tie_tolerance_m: float = 1e-8
    seed: int = 2026


@dataclass(frozen=True)
class Omega1Approximation:
    outer_polygon: np.ndarray
    inner_polygon: np.ndarray
    scenario_pool: np.ndarray
    circle_outer_gap_m: float


@dataclass(frozen=True)
class ScenarioEvaluation:
    radius_m: float
    diameter_m: float
    circle: Circle
    target: np.ndarray
    error_deg: float
    polygon: np.ndarray


@dataclass
class CandidateResult:
    method: str
    x: float
    y: float
    movement_m: float
    discretized_worst_radius_m: float
    worst_diameter_m: float
    worst_center_x: float
    worst_center_y: float
    worst_target_x: float
    worst_target_y: float
    worst_error_deg: float
    certified_min_distance_m: float
    certified_max_distance_m: float
    min_angle_sine: float
    meets_20m_on_discretization: bool

    def to_dict(self) -> dict[str, float | str | bool]:
        return asdict(self)


def _validate_config(config: Q2Config) -> None:
    numeric = asdict(config).values()
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("Q2配置包含非有限数")
    if not 0.0 < config.bearing_error_deg < 90.0:
        raise ValueError("bearing_error_deg必须在0到90度之间")
    if min(
        config.arena_radius_m,
        config.near_radius_m,
        config.guaranteed_radius_m,
        config.maximum_radius_m,
        config.candidate_step_m,
        config.refinement_step_m,
        config.tie_tolerance_m,
    ) <= 0.0:
        raise ValueError("半径、网格步长和容差必须为正数")
    if config.near_radius_m >= config.maximum_radius_m:
        raise ValueError("near半径必须小于最大接收半径")
    if config.disk_sides < 16:
        raise ValueError("圆的离散边数不得小于16")
    if min(config.scenario_pool_count, config.scenario_count, config.error_scenario_count) < 2:
        raise ValueError("场景数量不得小于2")


def _circle_polygon(
    center: tuple[float, float], radius: float, sides: int, circumscribed: bool
) -> np.ndarray:
    vertex_radius = radius / math.cos(math.pi / sides) if circumscribed else radius
    offset = math.pi / sides if circumscribed else 0.0
    angles = offset + np.arange(sides, dtype=float) * 2.0 * math.pi / sides
    return np.column_stack(
        (center[0] + vertex_radius * np.cos(angles), center[1] + vertex_radius * np.sin(angles))
    )


def _deduplicate_polygon(polygon: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    if len(polygon) == 0:
        return np.empty((0, 2), dtype=float)
    kept = [np.asarray(polygon[0], dtype=float)]
    for point in polygon[1:]:
        if np.linalg.norm(point - kept[-1]) > tol:
            kept.append(np.asarray(point, dtype=float))
    if len(kept) > 1 and np.linalg.norm(kept[0] - kept[-1]) <= tol:
        kept.pop()
    return np.asarray(kept, dtype=float)


def clip_polygon_halfplane(
    polygon: np.ndarray, halfplane: HalfPlane, tol: float = 1e-9
) -> np.ndarray:
    """用normal dot p >= offset裁剪逆时针凸多边形。"""
    if len(polygon) == 0:
        return np.empty((0, 2), dtype=float)
    normal = np.asarray(halfplane.normal, dtype=float)
    output: list[np.ndarray] = []
    previous = np.asarray(polygon[-1], dtype=float)
    previous_value = float(np.dot(normal, previous) - halfplane.offset)
    previous_inside = previous_value >= -tol
    for current_raw in polygon:
        current = np.asarray(current_raw, dtype=float)
        current_value = float(np.dot(normal, current) - halfplane.offset)
        current_inside = current_value >= -tol
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > 1e-15:
                output.append(previous + previous_value / denominator * (current - previous))
        if current_inside:
            output.append(current)
        previous, previous_value, previous_inside = current, current_value, current_inside
    return _deduplicate_polygon(np.asarray(output, dtype=float), tol)


def _intersect_convex_polygons(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    result = np.asarray(subject, dtype=float)
    for index, start in enumerate(clipper):
        end = clipper[(index + 1) % len(clipper)]
        edge = end - start
        normal = (-float(edge[1]), float(edge[0]))
        halfplane = HalfPlane(
            normal,
            normal[0] * float(start[0]) + normal[1] * float(start[1]),
        )
        result = clip_polygon_halfplane(result, halfplane)
        if len(result) == 0:
            break
    return result


def _clip_by_wedge(
    polygon: np.ndarray, sensor: tuple[float, float], bearing_deg: float, delta_deg: float
) -> np.ndarray:
    halfplanes, _ = build_geometry([Observation(*sensor, bearing_deg)], delta_deg)
    result = polygon
    for halfplane in halfplanes:
        result = clip_polygon_halfplane(result, halfplane)
    return result


def _point_in_convex_polygon(point: np.ndarray, polygon: np.ndarray, tol: float = 1e-9) -> bool:
    if len(polygon) < 3:
        return False
    for index, start in enumerate(polygon):
        edge = polygon[(index + 1) % len(polygon)] - start
        relative = point - start
        if edge[0] * relative[1] - edge[1] * relative[0] < -tol:
            return False
    return True


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    edge = end - start
    denominator = float(np.dot(edge, edge))
    if denominator <= 1e-24:
        return float(np.linalg.norm(point - start))
    fraction = float(np.dot(point - start, edge) / denominator)
    projection = start + min(1.0, max(0.0, fraction)) * edge
    return float(np.linalg.norm(point - projection))


def _point_polygon_distance(point: np.ndarray, polygon: np.ndarray) -> float:
    if _point_in_convex_polygon(point, polygon):
        return 0.0
    return min(
        _point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def _sample_polygon_boundary(polygon: np.ndarray, count: int) -> np.ndarray:
    ends = np.roll(polygon, -1, axis=0)
    lengths = np.linalg.norm(ends - polygon, axis=1)
    perimeter = float(lengths.sum())
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    positions = np.linspace(0.0, perimeter, count, endpoint=False)
    samples = []
    for position in positions:
        index = min(int(np.searchsorted(cumulative, position, side="right") - 1), len(polygon) - 1)
        fraction = (position - cumulative[index]) / lengths[index]
        samples.append(polygon[index] + fraction * (ends[index] - polygon[index]))
    return np.asarray(samples)


def _build_scenario_pool(config: Q2Config, inner_polygon: np.ndarray) -> np.ndarray:
    side_count = max(12, int(math.sqrt(config.scenario_pool_count)))
    angles = np.linspace(
        config.first_bearing_deg - config.bearing_error_deg,
        config.first_bearing_deg + config.bearing_error_deg,
        side_count,
    )
    radii = np.linspace(config.near_radius_m + 1e-6, config.maximum_radius_m, side_count)
    rr, aa = np.meshgrid(radii, np.deg2rad(angles), indexing="ij")
    polar = np.column_stack(
        (
            config.s1_x + rr.ravel() * np.cos(aa.ravel()),
            config.s1_y + rr.ravel() * np.sin(aa.ravel()),
        )
    )
    boundary = _sample_polygon_boundary(inner_polygon, max(side_count * 4, 80))
    points = np.vstack((polar, boundary, inner_polygon))
    s1 = np.array([config.s1_x, config.s1_y])
    arena = np.array([config.arena_x, config.arena_y])
    from_s1 = np.linalg.norm(points - s1, axis=1)
    from_arena = np.linalg.norm(points - arena, axis=1)
    inside = np.fromiter(
        (_point_in_convex_polygon(point, inner_polygon, 1e-7) for point in points), bool
    )
    mask = (
        (from_s1 > config.near_radius_m)
        & (from_s1 <= config.maximum_radius_m + 1e-8)
        & (from_arena <= config.arena_radius_m + 1e-8)
        & inside
    )
    points = points[mask]
    if len(points) == 0:
        raise ValueError("Omega1应用near排除后没有可用场景点")
    _, indices = np.unique(np.round(points, 9), axis=0, return_index=True)
    return points[np.sort(indices)]


def build_omega1(config: Q2Config) -> Omega1Approximation:
    """构造Omega1的内接、外接多边形及真实位置场景池。"""
    _validate_config(config)
    arena_center = (config.arena_x, config.arena_y)
    s1 = (config.s1_x, config.s1_y)
    arena_outer = _circle_polygon(arena_center, config.arena_radius_m, config.disk_sides, True)
    range_outer = _circle_polygon(s1, config.maximum_radius_m, config.disk_sides, True)
    outer = _clip_by_wedge(
        _intersect_convex_polygons(arena_outer, range_outer),
        s1,
        config.first_bearing_deg,
        config.bearing_error_deg,
    )
    arena_inner = _circle_polygon(arena_center, config.arena_radius_m, config.disk_sides, False)
    range_inner = _circle_polygon(s1, config.maximum_radius_m, config.disk_sides, False)
    inner = _clip_by_wedge(
        _intersect_convex_polygons(arena_inner, range_inner),
        s1,
        config.first_bearing_deg,
        config.bearing_error_deg,
    )
    if len(outer) < 3 or len(inner) < 3:
        raise ValueError("给定首测和竞技场条件下Omega1为空或退化")
    scenario_pool = _build_scenario_pool(config, inner)
    gap = max(config.arena_radius_m, config.maximum_radius_m) * (
        1.0 / math.cos(math.pi / config.disk_sides) - 1.0
    )
    return Omega1Approximation(outer, inner, scenario_pool, gap)


def candidate_grid(omega_outer: np.ndarray, step_m: float, search_radius_m: float) -> np.ndarray:
    anchor = omega_outer[0]
    mins = np.floor((anchor - search_radius_m) / step_m) * step_m
    maxs = np.ceil((anchor + search_radius_m) / step_m) * step_m
    xs = np.arange(mins[0], maxs[0] + step_m * 0.5, step_m)
    ys = np.arange(mins[1], maxs[1] + step_m * 0.5, step_m)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack((xx.ravel(), yy.ravel()))


def _long_axis_normal_seeds(omega_outer: np.ndarray, config: Q2Config) -> np.ndarray:
    center = omega_outer.mean(axis=0)
    covariance = np.cov((omega_outer - center).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, int(np.argmin(eigenvalues))]
    radii = np.arange(
        0.0,
        config.guaranteed_radius_m + config.candidate_step_m * 0.5,
        config.candidate_step_m / 2.0,
    )
    return np.asarray([center + sign * radius * normal for radius in radii for sign in (-1.0, 1.0)])


def distance_extrema(candidates: np.ndarray, omega_outer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.asarray([_point_polygon_distance(point, omega_outer) for point in candidates])
    maximum = np.max(
        np.linalg.norm(candidates[:, None, :] - omega_outer[None, :, :], axis=2), axis=1
    )
    return minimum, maximum


def feasible_candidates(
    candidates: np.ndarray, omega_outer: np.ndarray, config: Q2Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum, maximum = distance_extrema(candidates, omega_outer)
    mask = (maximum <= config.guaranteed_radius_m + 1e-9) & (
        minimum > config.near_radius_m + config.tie_tolerance_m
    )
    return candidates[mask], minimum[mask], maximum[mask]


def farthest_point_scenarios(points: np.ndarray, count: int) -> np.ndarray:
    count = min(max(1, count), len(points))
    chosen = [int(np.argmin(points[:, 0]))]
    nearest_sq = np.sum((points - points[chosen[0]]) ** 2, axis=1)
    while len(chosen) < count:
        index = int(np.argmax(nearest_sq))
        chosen.append(index)
        nearest_sq = np.minimum(nearest_sq, np.sum((points - points[index]) ** 2, axis=1))
    return points[np.asarray(chosen)]


def build_p2_outer_polygon(
    omega_outer: np.ndarray,
    sensor: np.ndarray,
    measured_bearing_deg: float,
    bearing_error_deg: float,
) -> np.ndarray:
    """显式构造有界外包络P2=Omega1_outer与W2的交集。"""
    return _clip_by_wedge(
        omega_outer,
        (float(sensor[0]), float(sensor[1])),
        measured_bearing_deg,
        bearing_error_deg,
    )


def _minimum_angle_sine(candidate: np.ndarray, scenarios: np.ndarray, s1: np.ndarray) -> float:
    first = scenarios - s1
    second = scenarios - candidate
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    cross = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    values = np.zeros(len(scenarios))
    valid = denominator > 1e-12
    values[valid] = cross[valid] / denominator[valid]
    return float(values.min())


def evaluate_candidate(
    candidate: np.ndarray,
    omega_outer: np.ndarray,
    scenarios: np.ndarray,
    errors_deg: Iterable[float],
    config: Q2Config,
) -> ScenarioEvaluation:
    worst: ScenarioEvaluation | None = None
    for target in scenarios:
        true_bearing = math.degrees(
            math.atan2(target[1] - candidate[1], target[0] - candidate[0])
        )
        for error in errors_deg:
            polygon = build_p2_outer_polygon(
                omega_outer,
                candidate,
                true_bearing + float(error),
                config.bearing_error_deg,
            )
            if len(polygon) == 0:
                current = ScenarioEvaluation(
                    math.inf,
                    math.inf,
                    Circle(math.nan, math.nan, math.inf),
                    target.copy(),
                    float(error),
                    polygon,
                )
            else:
                circle = minimum_enclosing_circle(polygon, seed=config.seed)
                diameter, _ = diameter_rotating_calipers(
                    [tuple(point) for point in polygon], 1e-10
                )
                current = ScenarioEvaluation(
                    circle.radius,
                    diameter,
                    circle,
                    target.copy(),
                    float(error),
                    polygon,
                )
            if worst is None or current.radius_m > worst.radius_m:
                worst = current
    assert worst is not None
    return worst


def _choose_main(
    candidates: np.ndarray,
    omega_outer: np.ndarray,
    scenarios: np.ndarray,
    errors: np.ndarray,
    config: Q2Config,
) -> np.ndarray:
    s1 = np.array([config.s1_x, config.s1_y])
    rows = []
    for candidate in candidates:
        evaluation = evaluate_candidate(candidate, omega_outer, scenarios, errors, config)
        rows.append((evaluation.radius_m, float(np.linalg.norm(candidate - s1)), candidate))
    best = min(row[0] for row in rows)
    contenders = [row for row in rows if row[0] <= best + config.tie_tolerance_m]
    return min(contenders, key=lambda row: row[1])[2]


def _choose_angle(candidates: np.ndarray, scenarios: np.ndarray, config: Q2Config) -> np.ndarray:
    s1 = np.array([config.s1_x, config.s1_y])
    scores = np.asarray([_minimum_angle_sine(point, scenarios, s1) for point in candidates])
    best = float(scores.max())
    contenders = candidates[np.isclose(scores, best, atol=1e-12, rtol=0.0)]
    return contenders[np.argmin(np.linalg.norm(contenders - s1, axis=1))]


def optimize(
    config: Q2Config,
    method: str = "M2-RR",
) -> tuple[CandidateResult | None, dict[str, float | int | bool], Omega1Approximation, np.ndarray]:
    if method not in {"M2-RR", "M2-ANGLE"}:
        raise ValueError("method必须是M2-RR或M2-ANGLE")
    omega = build_omega1(config)
    grid = candidate_grid(omega.outer_polygon, config.candidate_step_m, config.guaranteed_radius_m)
    seeds = _long_axis_normal_seeds(omega.outer_polygon, config)
    coarse = np.unique(np.round(np.vstack((grid, seeds)), 9), axis=0)
    feasible, _, _ = feasible_candidates(coarse, omega.outer_polygon, config)
    evidence: dict[str, float | int | bool] = {
        "omega_outer_vertices": len(omega.outer_polygon),
        "omega_inner_vertices": len(omega.inner_polygon),
        "scenario_pool_count": len(omega.scenario_pool),
        "circle_outer_gap_m": omega.circle_outer_gap_m,
        "coarse_grid_and_seed_count": len(coarse),
        "coarse_feasible_count": len(feasible),
        "fallback_triggered": len(feasible) == 0,
        "candidate_guarantee_uses_outer_omega": True,
    }
    if len(feasible) == 0:
        _, maximum = distance_extrema(coarse, omega.outer_polygon)
        evidence["minimum_certified_max_distance_m"] = float(maximum.min())
        evidence["minimum_reception_violation_m"] = float(
            max(0.0, maximum.min() - config.guaranteed_radius_m)
        )
        return None, evidence, omega, feasible

    scenarios = farthest_point_scenarios(omega.scenario_pool, config.scenario_count)
    errors = np.linspace(
        -config.bearing_error_deg,
        config.bearing_error_deg,
        config.error_scenario_count,
    )
    chosen = (
        _choose_main(feasible, omega.outer_polygon, scenarios, errors, config)
        if method == "M2-RR"
        else _choose_angle(feasible, scenarios, config)
    )
    offsets = np.arange(
        -config.candidate_step_m,
        config.candidate_step_m + config.refinement_step_m * 0.5,
        config.refinement_step_m,
    )
    local = np.asarray([chosen + (dx, dy) for dx in offsets for dy in offsets])
    local_feasible, _, _ = feasible_candidates(local, omega.outer_polygon, config)
    if len(local_feasible):
        chosen = (
            _choose_main(local_feasible, omega.outer_polygon, scenarios, errors, config)
            if method == "M2-RR"
            else _choose_angle(local_feasible, scenarios, config)
        )

    minimum, maximum = distance_extrema(chosen[None, :], omega.outer_polygon)
    evaluation = evaluate_candidate(chosen, omega.outer_polygon, scenarios, errors, config)
    s1 = np.array([config.s1_x, config.s1_y])
    result = CandidateResult(
        method=method,
        x=float(chosen[0]),
        y=float(chosen[1]),
        movement_m=float(np.linalg.norm(chosen - s1)),
        discretized_worst_radius_m=float(evaluation.radius_m),
        worst_diameter_m=float(evaluation.diameter_m),
        worst_center_x=float(evaluation.circle.x),
        worst_center_y=float(evaluation.circle.y),
        worst_target_x=float(evaluation.target[0]),
        worst_target_y=float(evaluation.target[1]),
        worst_error_deg=float(evaluation.error_deg),
        certified_min_distance_m=float(minimum[0]),
        certified_max_distance_m=float(maximum[0]),
        min_angle_sine=_minimum_angle_sine(chosen, scenarios, s1),
        meets_20m_on_discretization=bool(evaluation.radius_m <= 20.0),
    )
    evidence["local_feasible_count"] = len(local_feasible)
    evidence["scenario_count"] = len(scenarios)
    evidence["error_scenario_count"] = len(errors)
    evidence["error_endpoints_included"] = bool(
        math.isclose(float(errors[0]), -config.bearing_error_deg)
        and math.isclose(float(errors[-1]), config.bearing_error_deg)
    )
    return result, evidence, omega, feasible


def old_two_wedge_status(
    config: Q2Config, second_point: tuple[float, float], second_bearing_deg: float
) -> str:
    """仅用于回归诊断作废的W1与W2交集模型。"""
    observations = [
        Observation(config.s1_x, config.s1_y, config.first_bearing_deg),
        Observation(second_point[0], second_point[1], second_bearing_deg),
    ]
    return str(localize_source(observations, config.bearing_error_deg)["status"])
