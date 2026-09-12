"""Q4 V5离线实验：与V4同场景比较，重点报告10—12源。"""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path

from q4_client import OfflineMixedSimulatorClient
from q4_controller_v4 import Q4V4Controller
from q4_controller_v5 import Q4V5Controller
from q4_models import Q4Config
from q4_triangular_certificate import choose_mesh
from run_q4_experiment_v2 import clone, make_case

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "Q4" / "experiments" / "round5"


def execute(policy: str, case: dict, config: Q4Config, mesh) -> dict:
    client = OfflineMixedSimulatorClient(clone(case["sources"]), config.speed_mps)
    controller = Q4V5Controller(client, config) if policy == "Q4-V5-LOW-COUNT-JOINT" else Q4V4Controller(client, config)
    started = time.perf_counter()
    try:
        metrics = controller.run(mesh)
        status, error = "SUCCESS", ""
    except Exception as exc:
        metrics = dict(controller.metrics)
        metrics.update({"virtual_time_s": client.virtual_time_s, "wall_time_s": time.perf_counter() - started})
        status, error = "FAILED", f"{type(exc).__name__}: {exc}"
    return {"case": case["name"], "source_count": case["source_count"], "directional_count": case["directional_count"], "policy": policy, "status": status, "error": error, **metrics}


def aggregate(rows: list[dict]) -> dict:
    successful = [row for row in rows if row["status"] == "SUCCESS"]
    sources = sum(int(row["source_count"]) for row in successful)
    return {
        "run_count": len(rows),
        "successful_runs": len(successful),
        "pooled_seconds_per_source": sum(float(row["virtual_time_s"]) for row in successful) / max(1, sources),
        "mean_movement_m": sum(float(row["movement_m"]) for row in successful) / max(1, len(successful)),
        "mean_measurements": sum(float(row["measure_count"]) for row in successful) / max(1, len(successful)),
        "mean_clear_failures": sum(float(row["clear_failure"]) for row in successful) / max(1, len(successful)),
        "all_cleared_100_percent": len(successful) == len(rows) and all(int(row["cleared_count"]) == int(row["source_count"]) for row in successful),
        "all_runtime_source_count_unused": len(successful) == len(rows) and all(not bool(row.get("runtime_source_count_used", False)) for row in successful),
    }


def main() -> None:
    config, mesh = Q4Config(), choose_mesh(Q4Config())
    cases = [
        make_case("n10_all_directional_boundary", 10, 10, 802, True),
        make_case("n10_mixed_a", 10, 3, 905),
        make_case("n10_mixed_b", 10, 10, 901),
        make_case("n11_mixed", 11, 8, 803),
        make_case("n11_boundary", 11, 6, 504, True),
        make_case("n12_directional", 12, 11, 804),
        make_case("n12_mixed", 12, 6, 910),
        make_case("n13_boundary", 13, 10, 805, True),
        make_case("n14_mixed", 14, 9, 806),
        make_case("n15_directional", 15, 13, 807),
        make_case("n16_all_directional", 16, 16, 808, True),
    ]
    policies = ("Q4-V5-LOW-COUNT-JOINT", "Q4-V4-DYNAMIC")
    rows = [execute(policy, case, config, mesh) for case in cases for policy in policies]
    table = OUTPUT / "tables" / "policy_comparison.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    main_rows = [row for row in rows if row["policy"] == policies[0]]
    base_rows = [row for row in rows if row["policy"] == policies[1]]
    low_main = [row for row in main_rows if int(row["source_count"]) <= 12]
    low_base = [row for row in base_rows if int(row["source_count"]) <= 12]
    main_metrics, base_metrics = aggregate(main_rows), aggregate(base_rows)
    low_main_metrics, low_base_metrics = aggregate(low_main), aggregate(low_base)
    low_improvement = 100.0 * (1.0 - low_main_metrics["pooled_seconds_per_source"] / low_base_metrics["pooled_seconds_per_source"])
    overall_improvement = 100.0 * (1.0 - main_metrics["pooled_seconds_per_source"] / base_metrics["pooled_seconds_per_source"])
    metrics = {
        "q4_v5": main_metrics,
        "q4_v4": base_metrics,
        "low_count_10_to_12_v5": low_main_metrics,
        "low_count_10_to_12_v4": low_base_metrics,
        "low_count_improvement_percent": low_improvement,
        "overall_improvement_percent": overall_improvement,
        "target_seconds_per_source": 480.0,
        "target_met": main_metrics["all_cleared_100_percent"] and main_metrics["pooled_seconds_per_source"] <= 480.0,
    }
    metrics_path = OUTPUT / "metrics" / "q4_v5_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    all_success = all(row["status"] == "SUCCESS" for row in rows)
    summary = {
        "schema_version": 1,
        "question_id": "Q4",
        "approved_decision_id": "Q4-V5_low_count_rebuild_user_confirmed_2026-09-12",
        "method_roles": {"main": policies[0], "baseline": policies[1], "fallback": "Q4-V3 strict mesh"},
        "inputs": [str(ROOT / "B题.pdf"), str(ROOT / "Q4.qx_method_card.md"), str(ROOT / "code" / "Q4" / "q4_code_plan_v5.md")],
        "outputs": [str(table), str(metrics_path)],
        "seed": config.seed,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "execution_status": "SUCCESS" if all_success else "FAILED",
        "metrics": metrics,
        "output_degeneracy": {"failed_runs": sum(row["status"] != "SUCCESS" for row in rows), "zero_clear_runs": sum(int(row.get("clear_success", 0)) == 0 for row in rows)},
        "fallback_trigger_state": {"strict_mesh_fallback_runs": sum(bool(row.get("dynamic_fallback_used", False)) for row in main_rows), "wedge_fallback_runs": sum(bool(row.get("blind_wedge_used", False)) for row in main_rows)},
        "warnings": ["离线场景不代表官方随机分布。", "480秒/源是团队目标，并非题面阈值。", "10源极端边界场景可能不优于V4。"] + [row["error"] for row in rows if row["error"]],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "执行状态": summary["execution_status"],
        "V5全部场景汇总秒每源": round(main_metrics["pooled_seconds_per_source"], 3),
        "V4全部场景汇总秒每源": round(base_metrics["pooled_seconds_per_source"], 3),
        "V5低源数10至12汇总秒每源": round(low_main_metrics["pooled_seconds_per_source"], 3),
        "V4低源数10至12汇总秒每源": round(low_base_metrics["pooled_seconds_per_source"], 3),
        "低源数提升百分比": round(low_improvement, 2),
        "全部测试100%清除": main_metrics["all_cleared_100_percent"],
        "运行时读取真实源数": not main_metrics["all_runtime_source_count_unused"],
        "达到平均480秒每源": metrics["target_met"],
    }, ensure_ascii=False, indent=2))
    if not all_success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
