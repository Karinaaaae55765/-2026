"""运行Q2鲁棒第二检测点、交会角基线及收敛验证。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from q2_angle_baseline import run_angle_baseline
from q2_robust_second_measurement_optimizer import (
    Q2Config,
    build_omega1,
    build_p2_outer_polygon,
    evaluate_candidate,
    farthest_point_scenarios,
    old_two_wedge_status,
    optimize,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "Q2" / "experiments" / "round2"
ASSUMPTIONS = ROOT / "model_assumptions.md"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_candidate_region(config, omega, feasible, main, baseline, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    polygon = np.vstack((omega.outer_polygon, omega.outer_polygon[0]))
    ax.fill(polygon[:, 0], polygon[:, 1], color="#80b1d3", alpha=0.22, label=r"$\Omega_1$ 外包络")
    ax.scatter(omega.scenario_pool[:, 0], omega.scenario_pool[:, 1], s=4, color="#377eb8", alpha=0.25, label="目标位置场景")
    ax.scatter(feasible[:, 0], feasible[:, 1], s=30, facecolors="none", edgecolors="#4daf4a", label="严格候选点")
    ax.scatter(main["x"], main["y"], s=120, marker="*", color="#e41a1c", label="M2-RR")
    ax.scatter(baseline["x"], baseline["y"], s=70, marker="D", color="#984ea3", label="M2-ANGLE")
    ax.scatter(config.s1_x, config.s1_y, s=70, marker="^", color="#ff7f00", label=r"第一检测点 $S_1$")
    ax.set_xlabel("x（m，向东）")
    ax.set_ylabel("y（m，向北）")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_objective_map(config, omega, feasible, main, path: Path) -> None:
    scenarios = farthest_point_scenarios(omega.scenario_pool, config.scenario_count)
    errors = np.linspace(-config.bearing_error_deg, config.bearing_error_deg, config.error_scenario_count)
    values = [
        evaluate_candidate(candidate, omega.outer_polygon, scenarios, errors, config).radius_m
        for candidate in feasible
    ]
    fig, ax = plt.subplots(figsize=(5.2, 7.4), constrained_layout=True)
    scatter = ax.scatter(feasible[:, 0], feasible[:, 1], c=values, s=70, cmap="viridis")
    ax.scatter(main["x"], main["y"], s=120, marker="*", color="#e41a1c", label="M2-RR")
    ax.set_xlabel("x（m，向东）")
    ax.set_ylabel("y（m，向北）")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("离散最坏最小包围圆半径（m）")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _boundary_validation(config: Q2Config, main_result, main_evidence) -> list[dict]:
    omega = build_omega1(config)
    old_status = old_two_wedge_status(config, (-500.0, 0.0), 0.0)
    new_polygon = build_p2_outer_polygon(omega.outer_polygon, np.array([-500.0, 0.0]), 0.0, config.bearing_error_deg)
    cross_zero = build_omega1(replace(config, first_bearing_deg=359.8))
    empty_result, empty_evidence, _, _ = optimize(
        replace(config, guaranteed_radius_m=100.0, candidate_step_m=100.0, refinement_step_m=50.0, scenario_count=5, error_scenario_count=3),
        method="M2-RR",
    )
    rows = [
        {
            "case": "canonical_p2_replaces_two_wedges",
            "criterion": "old model unbounded and canonical P2 bounded",
            "observed": f"old={old_status}; new_vertices={len(new_polygon)}",
            "passed": old_status == "UNBOUNDED" and len(new_polygon) >= 3 and np.isfinite(new_polygon).all(),
        },
        {
            "case": "cross_zero_first_bearing",
            "criterion": "Omega1 remains nonempty at 359.8 degrees",
            "observed": f"outer_vertices={len(cross_zero.outer_polygon)}",
            "passed": len(cross_zero.outer_polygon) >= 3,
        },
        {
            "case": "strict_candidate_reception",
            "criterion": "max distance <= 1000 m",
            "observed": f"max={main_result.certified_max_distance_m:.9f}",
            "passed": main_result.certified_max_distance_m <= config.guaranteed_radius_m + 1e-8,
        },
        {
            "case": "strict_candidate_not_near",
            "criterion": "minimum distance > 5 m",
            "observed": f"min={main_result.certified_min_distance_m:.9f}",
            "passed": main_result.certified_min_distance_m > config.near_radius_m,
        },
        {
            "case": "bearing_error_endpoints",
            "criterion": "both -delta and +delta included",
            "observed": str(main_evidence["error_endpoints_included"]),
            "passed": bool(main_evidence["error_endpoints_included"]),
        },
        {
            "case": "empty_strict_candidate_set",
            "criterion": "100 m guarantee triggers fallback evidence",
            "observed": f"result_is_none={empty_result is None}; trigger={empty_evidence['fallback_triggered']}",
            "passed": empty_result is None and bool(empty_evidence["fallback_triggered"]),
        },
    ]
    return rows


def run(base: Q2Config | None = None) -> dict:
    started = time.perf_counter()
    base = base or Q2Config()
    assumption_hash = hashlib.sha256(ASSUMPTIONS.read_bytes()).hexdigest()
    levels = [
        ("coarse", replace(base, disk_sides=48, scenario_pool_count=400, candidate_step_m=150.0, refinement_step_m=50.0, scenario_count=15, error_scenario_count=3)),
        ("medium", replace(base, disk_sides=96, scenario_pool_count=625, candidate_step_m=100.0, refinement_step_m=25.0, scenario_count=25, error_scenario_count=5)),
        ("fine", replace(base, disk_sides=160, scenario_pool_count=900, candidate_step_m=100.0, refinement_step_m=25.0, scenario_count=41, error_scenario_count=7)),
    ]
    sensitivity_rows = []
    canonical = canonical_evidence = canonical_omega = canonical_feasible = None
    for level, config in levels:
        result, evidence, omega, feasible = optimize(config, method="M2-RR")
        row = {
            "level": level,
            "disk_sides": config.disk_sides,
            "scenario_count": config.scenario_count,
            "error_scenario_count": config.error_scenario_count,
            "candidate_step_m": config.candidate_step_m,
            "circle_outer_gap_m": evidence["circle_outer_gap_m"],
            "feasible_count": evidence["coarse_feasible_count"],
            "status": "OK" if result else "EMPTY_CANDIDATE_SET",
            "x_m": "" if result is None else result.x,
            "y_m": "" if result is None else result.y,
            "worst_radius_m": "" if result is None else result.discretized_worst_radius_m,
            "worst_diameter_m": "" if result is None else result.worst_diameter_m,
        }
        sensitivity_rows.append(row)
        if level == "fine":
            canonical, canonical_evidence, canonical_omega, canonical_feasible = result, evidence, omega, feasible
    if canonical is None:
        raise RuntimeError("细网格下严格候选域为空；请查看候选域违约证据")

    fine_config = levels[-1][1]
    baseline, baseline_evidence, _, _ = run_angle_baseline(fine_config)
    if baseline is None:
        raise RuntimeError("主方法可行但基线未找到候选点")

    comparison_rows = []
    for result in (canonical, baseline):
        comparison_rows.append(
            {
                "method": result.method,
                "x_m": result.x,
                "y_m": result.y,
                "movement_m": result.movement_m,
                "discretized_worst_radius_m": result.discretized_worst_radius_m,
                "worst_diameter_m": result.worst_diameter_m,
                "min_angle_sine": result.min_angle_sine,
                "certified_min_distance_m": result.certified_min_distance_m,
                "certified_max_distance_m": result.certified_max_distance_m,
                "within_20m": result.meets_20m_on_discretization,
            }
        )
    boundary_rows = _boundary_validation(fine_config, canonical, canonical_evidence)
    boundary_passed = all(bool(row["passed"]) for row in boundary_rows)

    table_dir = OUTPUT / "tables"
    figure_dir = OUTPUT / "figures"
    metrics_path = OUTPUT / "metrics" / "q2_metrics.json"
    comparison_path = table_dir / "candidate_comparison.csv"
    sensitivity_path = table_dir / "sensitivity.csv"
    boundary_path = table_dir / "boundary_validation.csv"
    candidate_figure = figure_dir / "q2_candidate_region.png"
    objective_figure = figure_dir / "q2_objective_map.png"
    _write_csv(comparison_path, comparison_rows)
    _write_csv(sensitivity_path, sensitivity_rows)
    _write_csv(boundary_path, boundary_rows)
    _plot_candidate_region(fine_config, canonical_omega, canonical_feasible, canonical.to_dict(), baseline.to_dict(), candidate_figure)
    _plot_objective_map(fine_config, canonical_omega, canonical_feasible, canonical.to_dict(), objective_figure)

    metrics = {
        "schema_version": 1,
        "question_id": "Q2",
        "decision_id": "q2_robust_radius_method",
        "model_assumptions_sha256": assumption_hash,
        "instance_type": "synthetic_demonstration_not_official_problem_data",
        "canonical_definition": "P2 = Omega1 intersection W2",
        "config": asdict(fine_config),
        "main": canonical.to_dict(),
        "baseline": baseline.to_dict(),
        "main_candidate_evidence": canonical_evidence,
        "baseline_candidate_evidence": baseline_evidence,
        "boundary_cases_passed": sum(bool(row["passed"]) for row in boundary_rows),
        "boundary_cases": len(boundary_rows),
        "sensitivity_levels": sensitivity_rows,
        "claim_scope": "candidate reception is certified by an outer polygon; target/error maximization is a converged finite discretization, not an analytic global optimum",
    }
    _write_json(metrics_path, metrics)
    elapsed = time.perf_counter() - started
    success = boundary_passed and np.isfinite(canonical.discretized_worst_radius_m)
    medium_row = sensitivity_rows[-2]
    fine_row = sensitivity_rows[-1]
    convergence = {
        "medium_to_fine_point_shift_m": float(
            np.hypot(fine_row["x_m"] - medium_row["x_m"], fine_row["y_m"] - medium_row["y_m"])
        ),
        "medium_to_fine_radius_change_m": float(
            abs(fine_row["worst_radius_m"] - medium_row["worst_radius_m"])
        ),
        "circle_outer_gap_reduction_m": float(
            medium_row["circle_outer_gap_m"] - fine_row["circle_outer_gap_m"]
        ),
    }
    summary = {
        "schema_version": 1,
        "question": "Q2",
        "round": "round2",
        "implementation_target": "python",
        "random_seed": fine_config.seed,
        "approved_decision_id": "q2_robust_radius_method",
        "model_assumptions_file": "model_assumptions.md",
        "model_assumptions_sha256": assumption_hash,
        "status": "success" if success else "failed",
        "execution_time_seconds": elapsed,
        "main_method": "M2-RR",
        "baseline_method": "M2-ANGLE",
        "methods": [
            {
                "method_id": "M2-RR",
                "role": "main",
                "script": "code/Q2/q2_robust_second_measurement_optimizer.py",
                "status": "success" if success else "failed",
                "metrics_summary": canonical.to_dict(),
                "warnings": ["真实位置与测向误差的最坏情形采用有限场景加密。"],
                "errors": [],
            },
            {
                "method_id": "M2-ANGLE",
                "role": "usable_baseline",
                "script": "code/Q2/q2_angle_baseline.py",
                "status": "success",
                "metrics_summary": baseline.to_dict(),
                "warnings": ["交会角仅为定位半径的代理指标。"],
                "errors": [],
            },
        ],
        "fallback_trigger": {
            "fallback_id": "M2-SLACK",
            "condition": "严格候选区域为空",
            "observed": False,
            "activated": False,
            "evidence": {"fine_coarse_feasible_count": canonical_evidence["coarse_feasible_count"]},
        },
        "main_result": canonical.to_dict(),
        "baseline_result": baseline.to_dict(),
        "comparison": {
            "radius_improvement_m": baseline.discretized_worst_radius_m - canonical.discretized_worst_radius_m,
            "main_within_20m": canonical.meets_20m_on_discretization,
            "boundary_validation_passed": boundary_passed,
            "convergence": convergence,
        },
        "output_degeneracy": {
            "strict_candidate_set_empty": False,
            "main_radius_finite": bool(np.isfinite(canonical.discretized_worst_radius_m)),
            "baseline_radius_finite": bool(np.isfinite(baseline.discretized_worst_radius_m)),
        },
        "outputs": {
            "metrics": metrics_path.relative_to(ROOT).as_posix(),
            "comparison": comparison_path.relative_to(ROOT).as_posix(),
            "sensitivity": sensitivity_path.relative_to(ROOT).as_posix(),
            "boundary_validation": boundary_path.relative_to(ROOT).as_posix(),
            "candidate_figure": candidate_figure.relative_to(ROOT).as_posix(),
            "objective_figure": objective_figure.relative_to(ROOT).as_posix(),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    _write_json(OUTPUT / "run_summary.json", summary)
    return summary


def print_chinese_summary(summary: dict) -> None:
    main = summary["main_result"]
    baseline = summary["baseline_result"]
    print("========== Q2 鲁棒第二检测点运行结果 ==========")
    print(f"运行状态：{'成功' if summary['status'] == 'success' else '失败'}")
    print("定位集合：P2 = Omega1 与 W2 的交集（旧定义已作废）")
    print("算例性质：合成演示，不是题目官方观测数据")
    print()
    print("主方法 M2-RR：")
    print(f"  第二检测点：({main['x']:.3f}, {main['y']:.3f}) m")
    print(f"  移动距离：{main['movement_m']:.3f} m")
    print(f"  离散最坏包围圆半径：{main['discretized_worst_radius_m']:.6f} m")
    print(f"  最坏区域直径：{main['worst_diameter_m']:.6f} m")
    print(f"  是否达到20 m：{'是' if main['meets_20m_on_discretization'] else '否'}")
    print()
    print("交会角基线 M2-ANGLE：")
    print(f"  第二检测点：({baseline['x']:.3f}, {baseline['y']:.3f}) m")
    print(f"  离散最坏包围圆半径：{baseline['discretized_worst_radius_m']:.6f} m")
    print(f"主方法相对基线改善：{summary['comparison']['radius_improvement_m']:.6f} m")
    print(f"边界验证：{'全部通过' if summary['comparison']['boundary_validation_passed'] else '存在失败'}")
    print("说明：连续最坏情形采用有限场景加密，不能表述为解析全局最优。")
    print(f"完整记录：{OUTPUT / 'run_summary.json'}")
    print("================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-x", type=float, default=0.0)
    parser.add_argument("--s1-y", type=float, default=0.0)
    parser.add_argument("--first-bearing-deg", type=float, default=0.0)
    parser.add_argument("--arena-x", type=float, default=0.0)
    parser.add_argument("--arena-y", type=float, default=0.0)
    parser.add_argument("--json", action="store_true", help="输出机器可读JSON")
    arguments = parser.parse_args()
    base = Q2Config(
        s1_x=arguments.s1_x,
        s1_y=arguments.s1_y,
        first_bearing_deg=arguments.first_bearing_deg,
        arena_x=arguments.arena_x,
        arena_y=arguments.arena_y,
    )
    result = run(base)
    if arguments.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_chinese_summary(result)
    if result["status"] != "success":
        raise SystemExit(1)
