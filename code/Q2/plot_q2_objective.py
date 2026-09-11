"""根据已保存的Q2运行摘要，独立生成目标函数诊断图。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from q2_robust_second_measurement_optimizer import (
    Q2Config,
    build_omega1,
    candidate_grid,
    classify_candidates,
    evaluate_candidate,
    farthest_point_scenarios,
)

ROOT = Path(__file__).resolve().parents[2]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _progress(message: str, started: float) -> None:
    print(f"[目标函数图 已运行 {time.perf_counter() - started:6.1f} 秒] {message}", flush=True)


def generate_objective_figure(quick: bool = False) -> Path:
    started = time.perf_counter()
    result_dir = ROOT / "results" / "Q2" / "experiments" / (
        "quick_check" if quick else "round3"
    )
    summary_path = result_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"未找到{summary_path}；请先运行Q2主算法生成摘要"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = Q2Config(**summary["config"])
    main = summary["main_result"]

    _progress("正在重建不确定区域和候选网格", started)
    omega = build_omega1(config)
    grid = candidate_grid(
        omega.outer_polygon, config.candidate_step_m, config.maximum_radius_m
    )
    classified = classify_candidates(grid, omega.scenario_pool, config)
    feasible = classified["signal"]
    if len(feasible) == 0:
        raise RuntimeError("已保存配置下没有响应候选点")

    scenarios = farthest_point_scenarios(omega.scenario_pool, config.scenario_count)
    errors = np.linspace(
        -config.bearing_error_deg,
        config.bearing_error_deg,
        config.error_scenario_count,
    )
    _progress(f"开始评价 {len(feasible)} 个候选点", started)
    values = []
    report_every = max(1, len(feasible) // 10)
    for index, point in enumerate(feasible, start=1):
        values.append(
            evaluate_candidate(
                point, omega.outer_polygon, scenarios, errors, config
            ).radius_m
        )
        if index % report_every == 0 or index == len(feasible):
            _progress(f"候选点评价进度：{index}/{len(feasible)}", started)

    fig, ax = plt.subplots(figsize=(5.4, 7.4), constrained_layout=True)
    scatter = ax.scatter(
        feasible[:, 0], feasible[:, 1], c=values, s=55, cmap="viridis"
    )
    ax.scatter(
        main["x"], main["y"], s=120, marker="*", color="#e41a1c", label="M2-RR"
    )
    ax.set_xlabel("x（m，向东）")
    ax.set_ylabel("y（m，向北）")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax).set_label("离散最坏最小包围圆半径（m）")
    output = result_dir / "figures" / "q2_objective_map.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    _progress(f"图片已生成：{output}", started)
    return output


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="读取quick_check摘要并生成低分辨率诊断图",
    )
    arguments = parser.parse_args()
    generate_objective_figure(quick=arguments.quick)
