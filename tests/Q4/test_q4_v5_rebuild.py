"""Q4 V5严格证书、真实联合调度与低源数回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q4 = ROOT / "code" / "Q4"
if str(Q4) not in sys.path:
    sys.path.insert(0, str(Q4))

from q4_client import OfflineMixedSimulatorClient  # noqa: E402
from q4_controller_v4 import Q4V4Controller  # noqa: E402
from q4_controller_v5 import Q4V5Controller  # noqa: E402
from q4_dynamic_certificate_v4 import certificate_status  # noqa: E402
from q4_models import Q4Config  # noqa: E402
from q4_state_filter_v5 import v5_certificate_template  # noqa: E402
from q4_triangular_certificate import choose_mesh  # noqa: E402
from run_q4_experiment_v2 import clone, make_case  # noqa: E402


class Q4V5RebuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Q4Config()
        self.mesh = choose_mesh(self.config)

    def run_policy(self, controller_type, case):
        client = OfflineMixedSimulatorClient(clone(case["sources"]), self.config.speed_mps)
        return controller_type(client, self.config).run(self.mesh)

    def test_25_point_template_is_strict(self) -> None:
        points = v5_certificate_template()
        status = certificate_status(points, self.config.arena_radius_m, self.config.minimum_radius_m)
        self.assertEqual(len(points), 25)
        self.assertTrue(status.complete)
        self.assertLess(status.maximum_intersecting_edge_m, self.config.minimum_radius_m)

    def test_unknown_source_count_and_joint_scheduler(self) -> None:
        case = make_case("n11", 11, 8, 803)
        result = self.run_policy(Q4V5Controller, case)
        self.assertEqual(result["cleared_count"], 11)
        self.assertEqual(result["cleared_count"] + result["absent_count"], 20)
        self.assertFalse(result["runtime_source_count_used"])
        self.assertTrue(result["strict_termination_verified"])
        self.assertGreater(result["joint_scheduler_calls"], 0)

    def test_low_count_pair_improves_over_v4(self) -> None:
        cases = [make_case("n11", 11, 8, 803), make_case("n12", 12, 11, 804)]
        v5 = sum(self.run_policy(Q4V5Controller, case)["virtual_time_s"] for case in cases)
        v4 = sum(self.run_policy(Q4V4Controller, case)["virtual_time_s"] for case in cases)
        self.assertLess(v5, v4)

    def test_16_source_regression_is_bounded(self) -> None:
        case = make_case("n16", 16, 16, 808, True)
        v5 = self.run_policy(Q4V5Controller, case)
        v4 = self.run_policy(Q4V4Controller, case)
        self.assertEqual(v5["cleared_count"], 16)
        self.assertLessEqual(v5["average_time_per_cleared_s"], 1.03 * v4["average_time_per_cleared_s"])


if __name__ == "__main__":
    unittest.main()
