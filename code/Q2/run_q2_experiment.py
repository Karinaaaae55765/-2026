"""运行Q2条件接收候选域、鲁棒第二点和交会角基线。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
    build_candidate_region_samples,
    build_omega1,
    build_p2_outer_polygon,
    candidate_metrics,
    classify_candidates,
    evaluate_candidate,
    old_two_wedge_status,
    optimize,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "Q2" / "experiments" / "round3"
QUICK_OUTPUT = ROOT / "results" / "Q2" / "experiments" / "quick_check"
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


def _local_coordinates(points: np.ndarray, config: Q2Config) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(config.first_bearing_deg)
    forward = np.array([math.cos(angle), math.sin(angle)])
    lateral = np.array([-math.sin(angle), math.cos(angle)])
    relative = points - np.array([config.s1_x, config.s1_y])
    return relative @ forward, relative @ lateral


def _region_rows(region: dict[str, np.ndarray], config: Q2Config) -> list[dict]:
    points = region["all"]
    forward, lateral = _local_coordinates(points, config)
    rows = []
    for index, point in enumerate(points):
        if not region["signal_mask"][index] and not region["bearing_mask"][index]:
            continue
        rows.append(
            {
                "x_m": float(point[0]),
                "y_m": float(point[1]),
                "forward_from_s1_m": float(forward[index]),
                "lateral_from_s1_m": float(lateral[index]),
                "in_signal_region": bool(region["signal_mask"][index]),
                "in_bearing_region": bool(region["bearing_mask"][index]),
                "sampled_signal_margin_m": float(region["signal_margin_m"][index]),
                "sampled_min_target_distance_m": float(region["minimum_distance_m"][index]),
                "sampled_max_target_distance_m": float(region["maximum_distance_m"][index]),
            }
        )
    return rows


def _region_summary(region: dict[str, np.ndarray], config: Q2Config) -> dict:
    signal = region["signal"]
    bearing = region["bearing"]
    step = config.candidate_region_step_m
    signal_forward, signal_lateral = _local_coordinates(signal, config)
    bearing_forward, bearing_lateral = _local_coordinates(bearing, config)

    def bounds(points: np.ndarray, forward: np.ndarray, lateral: np.ndarray) -> dict | None:
        if len(points) == 0:
            return None
        return {
            "x_min_m": float(points[:, 0].min()),
            "x_max_m": float(points[:, 0].max()),
            "y_min_m": float(points[:, 1].min()),
            "y_max_m": float(points[:, 1].max()),
            "forward_min_m": float(forward.min()),
            "forward_max_m": float(forward.max()),
            "lateral_min_m": float(lateral.min()),
            "lateral_max_m": float(lateral.max()),
        }

    return {
        "signal_region_definition": "for all (G,R) consistent with first success, distance(S2,G) <= R; equivalently distance(S2,G) <= max(1000,distance(S1,G)) for all G",
        "bearing_region_definition": "signal region plus distance(S2,G) > 5 m for every G",
        "representation": "uniform-grid approximation over the target scenario pool",
        "grid_step_m": step,
        "signal_region_nonempty": bool(len(signal)),
        "bearing_region_nonempty": bool(len(bearing)),
        "signal_grid_point_count": int(len(signal)),
        "bearing_grid_point_count": int(len(bearing)),
        "signal_approx_area_m2": float(len(signal) * step * step),
        "bearing_approx_area_m2": float(len(bearing) * step * step),
        "signal_bounds": bounds(signal, signal_forward, signal_lateral),
        "bearing_bounds": bounds(bearing, bearing_forward, bearing_lateral),
        "scope_note": "counts, areas and bounds approximate the continuous regions and depend on grid/scenario resolution",
    }


def _plot_regions(config, omega, region, main, baseline, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), constrained_layout=True)
    polygon = np.vstack((omega.outer_polygon, omega.outer_polygon[0]))
    for ax in axes:
        ax.fill(polygon[:, 0], polygon[:, 1], color="#80b1d3", alpha=0.22, label=r"$\Omega_1$ 外包络")
        ax.scatter(region["signal"][:, 0], region["signal"][:, 1], s=13, color="#a1d99b", label=r"响应候选域 $\mathcal{C}_{sig}$")
        ax.scatter(region["bearing"][:, 0], region["bearing"][:, 1], s=8, color="#238b45", label=r"示向候选域 $\mathcal{C}_{bearing}$")
        ax.scatter(main["x"], main["y"], s=115, marker="*", color="#e41a1c", label="M2-RR")
        ax.scatter(baseline["x"], baseline["y"], s=65, marker="D", color="#984ea3", label="M2-ANGLE")
        ax.scatter(config.s1_x, config.s1_y, s=65, marker="^", color="#ff7f00", label=r"第一检测点 $S_1$")
        ax.set_xlabel("x（m，向东）")
        ax.set_ylabel("y（m，向北）")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=9)
    combined = np.vstack((region["signal"], region["bearing"]))
    padding = 100.0
    axes[1].set_xlim(float(combined[:, 0].min() - padding), float(combined[:, 0].max() + padding))
    axes[1].set_ylim(float(combined[:, 1].min() - padding), float(combined[:, 1].max() + padding))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _boundary_checks(config: Q2Config, omega, main, evidence) -> list[dict]:
    test_point = np.array([[500.0, 800.0]])
    test_class = classify_candidates(test_point, omega.scenario_pool, config)
    _, _, old_max, _ = candidate_metrics(test_point, omega.scenario_pool, config)
    old_fixed_1000 = bool(old_max[0] <= config.guaranteed_radius_m)
    same_class = classify_candidates(
        np.array([[config.s1_x, config.s1_y]]), omega.scenario_pool, config
    )
    near_target = np.array([750.0, 0.0])
    near_eval = evaluate_candidate(
        near_target,
        omega.outer_polygon,
        np.array([near_target, [1200.0, 0.0]]),
        [-config.bearing_error_deg, config.bearing_error_deg],
        config,
    )
    old_status = old_two_wedge_status(config, (-500.0, 0.0), 0.0)
    new_polygon = build_p2_outer_polygon(
        omega.outer_polygon,
        np.array([-500.0, 0.0]),
        0.0,
        config.bearing_error_deg,
        config.maximum_radius_m,
        config.disk_sides,
    )
    cross_zero = build_omega1(replace(config, first_bearing_deg=359.8))
    return [
        {"case": "conditional_radius_recovers_valid_point", "observed": f"new={len(test_class['signal']) == 1}; old={old_fixed_1000}", "passed": len(test_class["signal"]) == 1 and not old_fixed_1000},
        {"case": "near_is_success_branch", "observed": f"near_scenarios={near_eval.near_scenario_count}", "passed": near_eval.near_scenario_count == 1},
        {"case": "same_location_excluded", "observed": f"signal_count={len(same_class['signal'])}", "passed": len(same_class["signal"]) == 0},
        {"case": "canonical_p2_bounded", "observed": f"old={old_status}; new_vertices={len(new_polygon)}", "passed": old_status == "UNBOUNDED" and len(new_polygon) >= 3},
        {"case": "cross_zero_first_bearing", "observed": f"outer_vertices={len(cross_zero.outer_polygon)}", "passed": len(cross_zero.outer_polygon) >= 3},
        {"case": "error_endpoints_included", "observed": str(evidence["error_endpoints_included"]), "passed": bool(evidence["error_endpoints_included"])},
        {"case": "main_signal_margin", "observed": f"margin={main.sampled_signal_margin_m:.9f}", "passed": main.sampled_signal_margin_m >= -config.tie_tolerance_m},
    ]


def _progress(message: str, started: float, enabled: bool) -> None:
    if enabled:
        print(f"[Q2 已运行 {time.perf_counter() - started:6.1f} 秒] {message}", flush=True)


def run(
    base: Q2Config | None = None,
    *,
    quick: bool = False,
    show_progress: bool = False,
) -> dict:
    started = time.perf_counter()
    base = base or Q2Config()
    output_dir = QUICK_OUTPUT if quick else OUTPUT
    assumption_hash = hashlib.sha256(ASSUMPTIONS.read_bytes()).hexdigest()
    if quick:
        levels = [
            ("quick", replace(base, disk_sides=48, scenario_pool_count=225, candidate_step_m=200.0, candidate_region_step_m=100.0, refinement_step_m=100.0, scenario_count=7, error_scenario_count=3))
        ]
    else:
        levels = [
            ("coarse", replace(base, disk_sides=48, scenario_pool_count=400, candidate_step_m=150.0, candidate_region_step_m=100.0, refinement_step_m=50.0, scenario_count=15, error_scenario_count=3)),
            ("medium", replace(base, disk_sides=96, scenario_pool_count=625, candidate_step_m=100.0, candidate_region_step_m=75.0, refinement_step_m=25.0, scenario_count=25, error_scenario_count=5)),
            ("fine", replace(base, disk_sides=160, scenario_pool_count=900, candidate_step_m=100.0, candidate_region_step_m=50.0, refinement_step_m=25.0, scenario_count=41, error_scenario_count=7)),
        ]
    _progress(
        "开始快速检查（低分辨率，不作为论文正式结果）" if quick else "开始正式计算（三档分辨率）",
        started,
        show_progress,
    )
    sensitivity = []
    canonical = canonical_evidence = canonical_omega = None
    for level, config in levels:
        _progress(f"正在计算主方法：{level} 分辨率", started, show_progress)
        result, evidence, omega, feasible = optimize(config, "M2-RR")
        sensitivity.append(
            {
                "level": level,
                "disk_sides": config.disk_sides,
                "target_scenarios": config.scenario_count,
                "error_scenarios": config.error_scenario_count,
                "candidate_step_m": config.candidate_step_m,
                "signal_candidate_count": evidence["coarse_signal_candidate_count"],
                "bearing_candidate_count": evidence["coarse_bearing_candidate_count"],
                "x_m": "" if result is None else result.x,
                "y_m": "" if result is None else result.y,
                "worst_radius_m": "" if result is None else result.discretized_worst_radius_m,
            }
        )
        _progress(
            f"{level} 分辨率完成；响应候选点 {evidence['coarse_signal_candidate_count']} 个",
            started,
            show_progress,
        )
        if level == levels[-1][0]:
            canonical, canonical_evidence, canonical_omega = result, evidence, omega
    if canonical is None:
        raise RuntimeError("条件接收候选域为空")
    fine_config = levels[-1][1]
    _progress("正在计算交会角基线", started, show_progress)
    baseline, baseline_evidence, _, _ = run_angle_baseline(fine_config)
    if baseline is None:
        raise RuntimeError("基线没有找到条件接收候选点")

    _progress("正在生成完整候选区域网格", started, show_progress)
    region = build_candidate_region_samples(canonical_omega, fine_config)
    region_summary = _region_summary(region, fine_config)
    region_rows = _region_rows(region, fine_config)
    checks = _boundary_checks(fine_config, canonical_omega, canonical, canonical_evidence)
    all_checks = all(bool(row["passed"]) for row in checks)
    comparison = []
    for result in (canonical, baseline):
        forward, lateral = _local_coordinates(np.array([[result.x, result.y]]), fine_config)
        comparison.append(
            {
                "method": result.method,
                "x_m": result.x,
                "y_m": result.y,
                "forward_from_s1_m": float(forward[0]),
                "lateral_from_s1_m": float(lateral[0]),
                "movement_m": result.movement_m,
                "discretized_worst_radius_m": result.discretized_worst_radius_m,
                "worst_diameter_m": result.worst_diameter_m,
                "sampled_signal_margin_m": result.sampled_signal_margin_m,
                "near_scenario_count": result.near_scenario_count,
                "bearing_scenario_count": result.bearing_scenario_count,
                "within_20m": result.meets_20m_on_discretization,
            }
        )

    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    paths = {
        "candidate_region": table_dir / "candidate_region.csv",
        "comparison": table_dir / "candidate_comparison.csv",
        "sensitivity": table_dir / "sensitivity.csv",
        "boundary": table_dir / "boundary_validation.csv",
        "candidate_figure": figure_dir / "q2_candidate_region.png",
        "metrics": output_dir / "metrics" / "q2_metrics.json",
        "summary": output_dir / "run_summary.json",
    }
    _progress("正在写入表格并生成候选区域图", started, show_progress)
    _write_csv(paths["candidate_region"], region_rows)
    _write_csv(paths["comparison"], comparison)
    _write_csv(paths["sensitivity"], sensitivity)
    _write_csv(paths["boundary"], checks)
    _plot_regions(fine_config, canonical_omega, region, canonical.to_dict(), baseline.to_dict(), paths["candidate_figure"])
    _progress("主算法计算完成；目标函数图请用独立脚本按需生成", started, show_progress)

    metrics = {
        "schema_version": 1,
        "question_id": "Q2",
        "decision_id": "q2_conditional_reception_2026-09-11",
        "model_assumptions_sha256": assumption_hash,
        "instance_type": "synthetic_demonstration_not_official_problem_data",
        "first_success_condition": "R >= distance(S1,G), 1000 <= R <= 1500",
        "candidate_regions": region_summary,
        "main": canonical.to_dict(),
        "baseline": baseline.to_dict(),
        "main_evidence": canonical_evidence,
        "baseline_evidence": baseline_evidence,
        "boundary_cases_passed": sum(bool(row["passed"]) for row in checks),
        "boundary_cases": len(checks),
        "sensitivity": sensitivity,
        "claim_scope": "candidate regions and worst-case objective use deterministic finite discretization; they are not analytic exact boundaries or a proof of the continuous global optimum",
    }
    _write_json(paths["metrics"], metrics)
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "question": "Q2",
        "round": "quick_check" if quick else "round3",
        "execution_mode": "quick" if quick else "formal",
        "implementation_target": "python",
        "random_seed": fine_config.seed,
        "config": asdict(fine_config),
        "approved_decision_id": "q2_conditional_reception_2026-09-11",
        "status": "success" if all_checks else "failed",
        "execution_time_seconds": elapsed,
        "model_assumptions_file": "model_assumptions.md",
        "model_assumptions_sha256": assumption_hash,
        "candidate_regions": region_summary,
        "main_result": canonical.to_dict(),
        "baseline_result": baseline.to_dict(),
        "methods": [
            {"method_id": "M2-RR", "role": "main", "script": "code/Q2/q2_robust_second_measurement_optimizer.py", "status": "success", "metrics_summary": canonical.to_dict(), "warnings": ["有限场景近似"], "errors": []},
            {"method_id": "M2-ANGLE", "role": "usable_baseline", "script": "code/Q2/q2_angle_baseline.py", "status": "success", "metrics_summary": baseline.to_dict(), "warnings": ["交会角是代理指标"], "errors": []},
        ],
        "comparison": {
            "radius_improvement_m": baseline.discretized_worst_radius_m - canonical.discretized_worst_radius_m,
            "main_within_20m": canonical.meets_20m_on_discretization,
            "boundary_validation_passed": all_checks,
        },
        "fallback_trigger": {"fallback_id": "M2-SLACK", "condition": "conditional signal region empty", "observed": False, "activated": False, "evidence": {"signal_candidate_count": canonical_evidence["coarse_signal_candidate_count"]}},
        "outputs": {key: value.relative_to(ROOT).as_posix() for key, value in paths.items() if key != "summary"},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
    }
    _write_json(paths["summary"], summary)
    _progress(f"全部完成，结果目录：{output_dir}", started, show_progress)
    return summary


def print_chinese_summary(summary: dict) -> None:
    region = summary["candidate_regions"]
    main = summary["main_result"]
    baseline = summary["baseline_result"]
    main_forward, main_lateral = _local_coordinates(
        np.array([[main["x"], main["y"]]]), Q2Config(**summary["config"])
    )
    print("========== Q2 鲁棒第二检测点运行结果 ==========")
    print(f"运行状态：{'成功' if summary['status'] == 'success' else '失败'}")
    print("第一次成功检测条件：R≥距离(S1,G)，且1000≤R≤1500 m")
    print("响应候选域：对每个可能目标，第二距离≤max(1000,第一距离)")
    print("示向候选域：响应候选域中再要求第二距离始终>5 m")
    print()
    print("候选区域（合成演示的网格近似）：")
    print(f"  响应候选域是否为空：{'否' if region['signal_region_nonempty'] else '是'}")
    print(f"  示向候选域是否为空：{'否' if region['bearing_region_nonempty'] else '是'}")
    print(f"  网格步长：{region['grid_step_m']:.3f} m")
    print(f"  响应候选点：{region['signal_grid_point_count']} 个，近似面积：{region['signal_approx_area_m2']:.3f} 平方米")
    print(f"  示向候选点：{region['bearing_grid_point_count']} 个，近似面积：{region['bearing_approx_area_m2']:.3f} 平方米")
    print(f"  运行模式：{'快速检查' if summary['execution_mode'] == 'quick' else '正式计算'}")
    print(f"  完整候选点文件：{ROOT / summary['outputs']['candidate_region']}")
    print("  注：面积和边界依赖网格分辨率，不是解析精确值。")
    print()
    print("主方法 M2-RR：")
    print(f"  第二检测点：({main['x']:.3f}, {main['y']:.3f}) m")
    print(f"  相对首测方向：前向{main_forward[0]:.3f} m，侧向{main_lateral[0]:.3f} m")
    print(f"  离散最坏包围圆半径：{main['discretized_worst_radius_m']:.6f} m")
    print(f"  条件接收最小余量：{main['sampled_signal_margin_m']:.6f} m")
    print(f"  是否达到20 m：{'是' if main['meets_20m_on_discretization'] else '否'}")
    print()
    print(f"交会角基线半径：{baseline['discretized_worst_radius_m']:.6f} m")
    print(f"主方法相对基线改善：{summary['comparison']['radius_improvement_m']:.6f} m")
    print(f"边界验证：{'全部通过' if summary['comparison']['boundary_validation_passed'] else '存在失败'}")
    print("说明：当前数值为合成演示，题目未给具体S1和首测示向度。")
    print("================================================")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-x", type=float, default=0.0)
    parser.add_argument("--s1-y", type=float, default=0.0)
    parser.add_argument("--first-bearing-deg", type=float, default=0.0)
    parser.add_argument("--arena-x", type=float, default=0.0)
    parser.add_argument("--arena-y", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="运行低分辨率快速检查并写入独立quick_check目录",
    )
    args = parser.parse_args()
    config = Q2Config(s1_x=args.s1_x, s1_y=args.s1_y, first_bearing_deg=args.first_bearing_deg, arena_x=args.arena_x, arena_y=args.arena_y)
    result = run(config, quick=args.quick, show_progress=not args.json)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_chinese_summary(result)
    if result["status"] != "success":
        raise SystemExit(1)
