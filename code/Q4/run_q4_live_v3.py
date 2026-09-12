"""Q4 V3正式入口：源数未知，只有严格频道证书完成后才退出。"""

from __future__ import annotations

import argparse
import json
import sys

from q4_client import SimulatorClient
from q4_controller_v3 import Q4V3Controller
from q4_models import Q4Config
from q4_task_scheduler_v3 import certificate_priority_order
from q4_triangular_certificate import choose_mesh, validate_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Q4未知源数严格证书程序（第三版）")
    parser.add_argument("--robot-id", help="参赛账号，仅作为接口参数，不写入代码或结果")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--confirm-live", action="store_true")
    arguments = parser.parse_args()

    config = Q4Config()
    mesh = choose_mesh(config)
    certificate = validate_mesh(mesh, config)
    route = certificate_priority_order((0.0, 0.0), mesh, config)
    if not certificate["edge_strictly_below_minimum_radius"] or set(route) != set(range(len(mesh.vertices))):
        raise SystemExit("Q4 V3三角证书自检失败")
    if arguments.self_check:
        print(json.dumps({
            "程序版本": "Q4_Robot_v3",
            "自检通过": True,
            "是否访问模拟器": False,
            "运行时是否使用干扰源总数": False,
            "三角网格顶点数": len(mesh.vertices),
            "三角单元数": len(mesh.triangles),
            "最大三角边长_米": round(certificate["maximum_edge_m"], 6),
            "严格停止条件": "20个频道全部已清除或由全域三角证书证明不存在",
            "正式运行示例": ".\\Q4_Robot_v3.exe --robot-id <账号> --confirm-live",
        }, ensure_ascii=False, indent=2))
        return
    if not arguments.confirm_live:
        raise SystemExit("未提供 --confirm-live：程序未调用模拟器")
    if not arguments.robot_id:
        raise SystemExit("正式运行缺少 --robot-id")
    print("[Q4 V3] 未知源数证书自检通过，正在连接仿真器……", flush=True)
    result = Q4V3Controller(SimulatorClient(arguments.base_url, arguments.robot_id), config).run(mesh)
    print(json.dumps({
        "检测次数": result["measure_count"],
        "切换频道次数": result["switch_count"],
        "成功清除数量": result["clear_success"],
        "清除失败数量": result["clear_failure"],
        "总移动距离_米": round(result["movement_m"], 6),
        "搜索移动距离_米": round(result["search_movement_m"], 6),
        "定位清除移动距离_米": round(result["localization_clear_movement_m"], 6),
        "访问三角顶点数": result["triangle_vertices_visited"],
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
