"""Run deterministic Q1 validation cases and save round-1 evidence."""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

from q1_geometry import (
    Observation,
    convex_hull,
    diameter_bruteforce,
    diameter_circle_report,
    diameter_rotating_calipers,
    localize_source,
    plot_localization,
    point_in_all_wedges,
)

SEED = 2026
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUND_DIR = PROJECT_ROOT / "results" / "Q1" / "experiments" / "round1"


def bearing(source: tuple[float, float], detector: tuple[float, float]) -> float:
    return math.degrees(math.atan2(source[1] - detector[1], source[0] - detector[0])) % 360.0


def observations_for(source: tuple[float, float], detectors: list[tuple[float, float]]) -> list[Observation]:
    return [Observation(x, y, bearing(source, (x, y))) for x, y in detectors]


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    tables_dir = ROUND_DIR / "tables"
    metrics_dir = ROUND_DIR / "metrics"
    figures_dir = ROUND_DIR / "figures"
    for directory in (tables_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    true_source = (300.0, 200.0)
    bounded_observations = observations_for(
        true_source,
        [(-600.0, -200.0), (900.0, -400.0), (700.0, 900.0), (-500.0, 700.0)],
    )
    cross_zero_source = (1000.0, 0.0)
    cross_zero_observations = observations_for(
        cross_zero_source,
        [(0.0, 10.0), (0.0, -10.0), (1600.0, 400.0), (1600.0, -400.0)],
    )

    cases = {
        "bounded_consistency": (bounded_observations, true_source, "OK"),
        "cross_zero": (cross_zero_observations, cross_zero_source, "OK"),
        "unbounded_single_wedge": ([Observation(0.0, 0.0, 20.0)], None, "UNBOUNDED"),
        "empty_opposed_shifted": (
            [Observation(0.0, 0.0, 0.0), Observation(0.0, 10.0, 180.0)],
            None,
            "EMPTY",
        ),
        "degenerate_shared_apex": (
            [Observation(0.0, 0.0, 0.0), Observation(0.0, 0.0, 180.0)],
            (0.0, 0.0),
            "DEGENERATE",
        ),
    }

    rows: list[dict[str, object]] = []
    results: dict[str, dict] = {}
    for name, (observations, source, expected_status) in cases.items():
        result = localize_source(observations)
        results[name] = result
        contains_source = (
            None if source is None else point_in_all_wedges(source, observations, tol=1e-7)
        )
        passed = result["status"] == expected_status and contains_source is not False
        rows.append(
            {
                "case": name,
                "expected_status": expected_status,
                "actual_status": result["status"],
                "num_vertices": len(result["vertices_ccw"]),
                "diameter_m": "" if result["diameter"] is None else f"{result['diameter']:.12g}",
                "covers_region": "" if result["diameter_circle"] is None else result["diameter_circle"]["covers_region"],
                "true_source_feasible": "" if contains_source is None else contains_source,
                "passed": passed,
            }
        )

    # The bearing error may attain either endpoint and must remain feasible.
    boundary_source = (500.0, 250.0)
    boundary_detectors = [(-400.0, -300.0), (900.0, -500.0), (1000.0, 900.0), (-300.0, 800.0)]
    boundary_observations = []
    for index, detector in enumerate(boundary_detectors):
        exact = bearing(boundary_source, detector)
        signed_error = 1.0 if index % 2 == 0 else -1.0
        boundary_observations.append(Observation(*detector, exact + signed_error))
    boundary_result = localize_source(boundary_observations)
    boundary_feasible = point_in_all_wedges(boundary_source, boundary_observations, tol=1e-7)
    boundary_passed = boundary_result["status"] in {"OK", "DEGENERATE"} and boundary_feasible
    rows.append(
        {
            "case": "error_endpoint_inclusive",
            "expected_status": "OK_OR_DEGENERATE",
            "actual_status": boundary_result["status"],
            "num_vertices": len(boundary_result["vertices_ccw"]),
            "diameter_m": "" if boundary_result["diameter"] is None else f"{boundary_result['diameter']:.12g}",
            "covers_region": boundary_result["diameter_circle"]["covers_region"],
            "true_source_feasible": boundary_feasible,
            "passed": boundary_passed,
        }
    )

    # A diameter-D set need not fit in the circle having a diameter pair as diameter.
    triangle = [(0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3.0) / 2.0)]
    triangle_diameter, triangle_pair = diameter_rotating_calipers(triangle)
    assert triangle_pair is not None
    triangle_circle = diameter_circle_report(triangle, triangle_pair, 1e-12)
    triangle_passed = not triangle_circle["covers_region"]
    rows.append(
        {
            "case": "equilateral_triangle_counterexample",
            "expected_status": "NOT_COVERED",
            "actual_status": "NOT_COVERED" if triangle_passed else "COVERED",
            "num_vertices": 3,
            "diameter_m": f"{triangle_diameter:.12g}",
            "covers_region": triangle_circle["covers_region"],
            "true_source_feasible": "",
            "passed": triangle_passed,
        }
    )

    # Risk-targeted randomized comparison of rotating calipers and the baseline.
    rng = np.random.default_rng(SEED)
    max_difference = 0.0
    random_cases = 250
    for _ in range(random_cases):
        points = [tuple(map(float, row)) for row in rng.normal(size=(40, 2))]
        hull = convex_hull(points)
        main_diameter, _ = diameter_rotating_calipers(hull, 1e-12)
        baseline_diameter, _ = diameter_bruteforce(hull)
        max_difference = max(max_difference, abs(main_diameter - baseline_diameter))
    random_passed = max_difference <= 1e-10

    table_path = tables_dir / "validation_cases.csv"
    with table_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure_path = figures_dir / "q1_example.png"
    plot_localization(bounded_observations, results["bounded_consistency"], figure_path)

    all_deterministic_passed = all(bool(row["passed"]) for row in rows)
    metrics = {
        "schema_version": 1,
        "random_seed": SEED,
        "deterministic_cases": len(rows),
        "deterministic_cases_passed": sum(bool(row["passed"]) for row in rows),
        "all_deterministic_passed": all_deterministic_passed,
        "random_convex_polygons_compared": random_cases,
        "max_main_baseline_diameter_difference": max_difference,
        "random_comparison_passed": random_passed,
        "bounded_example": {
            "num_vertices": len(results["bounded_consistency"]["vertices_ccw"]),
            "diameter_m": results["bounded_consistency"]["diameter"],
            "diameter_circle_covers": results["bounded_consistency"]["diameter_circle"]["covers_region"],
        },
        "equilateral_triangle": {
            "diameter": triangle_diameter,
            "diameter_circle_radius": triangle_circle["radius"],
            "max_vertex_distance_from_circle_center": triangle_circle["max_vertex_distance"],
            "covers_region": triangle_circle["covers_region"],
        },
        "output_degeneracy": {
            "empty_detected": results["empty_opposed_shifted"]["status"] == "EMPTY",
            "unbounded_detected": results["unbounded_single_wedge"]["status"] == "UNBOUNDED",
            "degenerate_detected": results["degenerate_shared_apex"]["status"] == "DEGENERATE",
        },
    }
    metrics_path = metrics_dir / "q1_metrics.json"
    save_json(metrics_path, metrics)

    elapsed = time.perf_counter() - started
    success = all_deterministic_passed and random_passed
    run_summary = {
        "schema_version": 1,
        "question": "Q1",
        "round": "round1",
        "implementation_target": "python",
        "random_seed": SEED,
        "approved_decision_id": "q1_model_assumptions_2026-09-10",
        "methods": [
            {
                "method_id": "wedge_intersection_rotating_calipers",
                "role": "main",
                "script": "code/Q1/q1_geometry.py",
                "status": "success" if success else "failed",
                "execution_time_seconds": elapsed,
                "input_files": ["model_assumptions.md"],
                "output_files": [str(table_path.relative_to(PROJECT_ROOT)), str(metrics_path.relative_to(PROJECT_ROOT))],
                "figure_files": [str(figure_path.relative_to(PROJECT_ROOT))],
                "metrics_summary": {
                    "deterministic_cases_passed": metrics["deterministic_cases_passed"],
                    "deterministic_cases": metrics["deterministic_cases"],
                    "random_comparison_passed": random_passed,
                },
                "warnings": ["No official numeric Q1 observation set was supplied; validation uses deterministic constructed cases."],
                "errors": [] if success else ["One or more validation checks failed."],
            },
            {
                "method_id": "vertex_pair_enumeration",
                "role": "usable_baseline",
                "script": "code/Q1/q1_baseline.py",
                "status": "success" if random_passed else "failed",
                "execution_time_seconds": elapsed,
                "input_files": [],
                "output_files": [str(metrics_path.relative_to(PROJECT_ROOT))],
                "figure_files": [],
                "metrics_summary": {"max_diameter_difference": max_difference},
                "warnings": [],
                "errors": [] if random_passed else ["Diameter mismatch detected."],
            },
        ],
        "comparison": {
            "metric": "absolute diameter difference",
            "cases": random_cases,
            "maximum_difference": max_difference,
            "passed": random_passed,
        },
        "fallback_trigger": {
            "fallback_id": None,
            "condition": None,
            "observed": False,
            "evidence": None,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    summary_path = ROUND_DIR / "run_summary.json"
    save_json(summary_path, run_summary)

    print(json.dumps({"success": success, "metrics": metrics, "run_summary": str(summary_path)}, ensure_ascii=False, indent=2))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

