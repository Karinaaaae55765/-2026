"""Q4 V5仿真器入口；所有摘要字段使用中文。"""

from __future__ import annotations

import argparse
import json
import sys

from q4_client import SimulatorClient
from q4_controller_v5 import Q4V5Controller
from q4_dynamic_certificate_v4 import certificate_status
from q4_models import Q4Config
from q4_state_filter_v5 import v5_certificate_template
from q4_triangular_certificate import choose_mesh, validate_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Q4低源数联合优化程序（第五版）")
    parser.add_argument("--robot-id", help="参赛账号，仅用于本次接口请求")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--confirm-live", action="store_true")
    arguments = parser.parse_args()
    config = Q4Config()
    mesh = choose_mesh(config)
    mesh_check = validate_mesh(mesh, config)
    template_check = certificate_status(v5_certificate_template(), config.arena_radius_m, config.minimum_radius_m)
    if not mesh_check["edge_strictly_below_minimum_radius"] or not template_check.complete:
        raise SystemExit("Q4 V5严格几何证书自检失败")
    if arguments.self_check:
        print(json.dumps({
            "程序版本": "Q4_Robot_v5",
            "自检通过": True,
            "是否访问模拟器": False,
            "运行时是否使用干扰源总数": False,
            "严格候选点数": len(v5_certificate_template()),
            "最大相交三角边_米": round(template_check.maximum_intersecting_edge_m, 6),
            "停止条件": "清除16源，或严格证书证明其余频道不存在且全部已发现源清除",
            "正式命令": ".\\Q4_Robot_v5.exe --robot-id <账号> --confirm-live",
        }, ensure_ascii=False, indent=2))
        return
    if not arguments.confirm_live:
        raise SystemExit("未提供 --confirm-live：程序未调用模拟器")
    if not arguments.robot_id:
        raise SystemExit("正式运行缺少 --robot-id")
    print("[Q4 V5] 低源数联合证书自检通过，正在连接仿真器……", flush=True)
    result = Q4V5Controller(SimulatorClient(arguments.base_url, arguments.robot_id), config).run(mesh)
    print(json.dumps({
        "检测次数": result["measure_count"],
        "切换频道次数": result["switch_count"],
        "成功清除数量": result["clear_success"],
        "清除失败数量": result["clear_failure"],
        "总移动距离_米": round(result["movement_m"], 6),
        "搜索移动距离_米": round(result["search_movement_m"], 6),
        "定位清除移动距离_米": round(result["localization_clear_movement_m"], 6),
        "严格证书访问点数": result["dynamic_points_visited"],
        "严格证书是否完成": result["dynamic_certificate_complete"],
        "是否使用后备网格": result["dynamic_fallback_used"],
        "联合调度调用次数": result["joint_scheduler_calls"],
        "Q2补测次数": result["q2_probe_measurements"],
        "Q2补测正响应次数": result["q2_probe_positive"],
        "是否使用严格楔形后备": result["blind_wedge_used"],
        "按16源上界排除频道数": result["absent_by_cardinality"],
        "不存在频道数": result["absent_count"],
        "运行时是否使用源数": result["runtime_source_count_used"],
        "严格停止条件是否满足": result["strict_termination_verified"],
        "最终虚拟总时间_秒": round(result["virtual_time_s"], 6),
        "平均每清除一个源时间_秒": round(result["average_time_per_cleared_s"], 6),
        "程序实际运行时间_秒": round(result["wall_time_s"], 6),
        "调度器版本": result["scheduler_version"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
