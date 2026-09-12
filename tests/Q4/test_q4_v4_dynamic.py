"""Q4 V4动态几何与未知源数测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q4 = ROOT / "code" / "Q4"
if str(Q4) not in sys.path:
    sys.path.insert(0, str(Q4))

from q4_client import OfflineMixedSimulatorClient, OfflineSource  # noqa: E402
from q4_controller_v4 import Q4V4Controller  # noqa: E402
from q4_dynamic_certificate_v4 import certificate_status, initial_certificate_candidates  # noqa: E402
from q4_models import Q4Config  # noqa: E402
from q4_triangular_certificate import choose_mesh  # noqa: E402


class LeakyClient(OfflineMixedSimulatorClient):
    def enter(self) -> dict:
        result = super().enter()
        result["jammer_count"] = 16
        return result


class DynamicCertificateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Q4Config()

    def test_outer_and_inner_template_has_valid_hull(self) -> None:
        points = [(0.0, 0.0)] + initial_certificate_candidates(self.config.arena_radius_m)
        status = certificate_status(points, self.config.arena_radius_m, self.config.minimum_radius_m)
        self.assertTrue(status.hull_covers_arena)
        self.assertGreater(status.intersecting_triangles, 0)
        self.assertTrue(status.complete)
        self.assertLess(status.maximum_intersecting_edge_m, self.config.minimum_radius_m)

    def test_unknown_count_and_strict_clear(self) -> None:
        sources = [OfflineSource(channel, (-700.0 + 110.0 * channel, -500.0 + 75.0 * channel), 1350.0, "D", 23.0 * channel) for channel in range(1, 11)]
        controller = Q4V4Controller(LeakyClient(sources), self.config)
        result = controller.run(choose_mesh(self.config))
        self.assertFalse(result["runtime_source_count_used"])
        self.assertEqual(result["cleared_count"], 10)
        self.assertEqual(result["cleared_count"] + result["absent_count"], 20)
        self.assertTrue(result["strict_termination_verified"])


if __name__ == "__main__":
    unittest.main()
