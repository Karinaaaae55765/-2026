"""Q4 V3未知源数离线验证，并与合法基线直接比较。"""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path

from q4_baseline import Q4BaselineController
from q4_client import OfflineMixedSimulatorClient
from q4_controller_v3 import Q4V3Controller
from q4_models import Q4Config
from q4_triangular_certificate import choose_mesh, validate_mesh
from run_q4_experiment_v2 import clone, make_case

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "Q4" / "experiments" / "round3"


def execute(policy: str, case: dict, config: Q4Config, mesh) -> dict:
    client = OfflineMixedSimulatorClient(clone(case["sources"]), config.speed_mps)
    controller = Q4V3Controller(client, config) if policy == "Q4-V3-UNKNOWN" else Q4BaselineController(client, config)
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
        "all_cleared_100_percent": len(successful) == len(rows) and all(int(row["cleared_count"]) == int(row["source_count"]) for row in successful),
        "all_runtime_source_count_unused": len(successful) == len(rows) and all(not bool(row.get("runtime_source_count_used", False)) for row in successful),
    }


def main() -> None:
    config = Q4Config()
    mesh = choose_mesh(config)
    certificate = validate_mesh(mesh, config)
    cases = [
        make_case("n10_all_directional_boundary", 10, 10, 702, True),
        make_case("n11_mixed", 11, 6, 703),
        make_case("n12_directional", 12, 11, 704),
        make_case("n13_boundary", 13, 9, 705, True),
        make_case("n14_mixed", 14, 8, 706),
        make_case("n15_all_directional", 15, 15, 707),
        make_case("n16_mixed_boundary", 16, 10, 708, True),
    ]
    rows = [execute(policy, case, config, mesh) for case in cases for policy in ("Q4-V3-UNKNOWN", "Q4-BASE")]
    table = OUTPUT / "tables" / "policy_comparison.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    main_metrics = aggregate([row for row in rows if row["policy"] == "Q4-V3-UNKNOWN"])
    base_metrics = aggregate([row for row in rows if row["policy"] == "Q4-BASE"])
    improvement = 100.0 * (1.0 - main_metrics["pooled_seconds_per_source"] / base_metrics["pooled_seconds_per_source"])
    metrics = {
        "q4_v3_unknown": main_metrics,
        "q4_base": base_metrics,
        "improvement_vs_base_percent": improvement,
        "target_seconds_per_source": 480.0,
        "target_met": main_metrics["all_cleared_100_percent"] and main_metrics["pooled_seconds_per_source"] <= 480.0,
    }
    metrics_path = OUTPUT / "metrics" / "q4_v3_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    all_success = all(row["status"] == "SUCCESS" for row in rows)
    summary = {
        "schema_version": 1,
        "question_id": "Q4",
        "approved_decision_id": "Q4-V3_unknown_count_user_confirmed_2026-09-12",
        "method_roles": {"main": "Q4-V3-UNKNOWN", "baseline": "Q4-BASE"},
        "inputs": [str(ROOT / "B题.pdf"), str(ROOT / "Q4.qx_method_card.md"), str(ROOT / "code" / "Q4" / "q4_code_plan_v3.md")],
        "outputs": [str(table), str(metrics_path)],
        "seed": config.seed,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "execution_status": "SUCCESS" if all_success else "FAILED",
        "metrics": metrics,
        "output_degeneracy": {"failed_runs": sum(row["status"] != "SUCCESS" for row in rows), "zero_clear_runs": sum(int(row.get("clear_success", 0)) == 0 for row in rows)},
        "fallback_trigger_state": {"source_count_available": False, "full_certificate_required": True},
        "warnings": ["离线场景只验证算法逻辑，不代表官方随机分布。", "480秒/源是团队目标，不是题面阈值。"] + [row["error"] for row in rows if row["error"]],
        "mesh_certificate": certificate,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "执行状态": summary["execution_status"],
        "V3未知源数汇总秒每源": round(main_metrics["pooled_seconds_per_source"], 3),
        "严格基线汇总秒每源": round(base_metrics["pooled_seconds_per_source"], 3),
        "相对基线提升百分比": round(improvement, 2),
        "全部场景100%清除": main_metrics["all_cleared_100_percent"],
        "运行时使用源数": not main_metrics["all_runtime_source_count_unused"],
        "达到480秒每源": metrics["target_met"],
    }, ensure_ascii=False, indent=2))
    if not all_success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
