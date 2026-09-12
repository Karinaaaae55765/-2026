"""Q4 V3正式口径回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q4 = ROOT / "code" / "Q4"
if str(Q4) not in sys.path:
    sys.path.insert(0, str(Q4))

from q4_client import OfflineMixedSimulatorClient, OfflineSource  # noqa: E402
from q4_controller_v3 import Q4V3Controller  # noqa: E402
from q4_models import ChannelStatus, Q4Config  # noqa: E402
from q4_task_scheduler_v3 import certificate_priority_order  # noqa: E402
from q4_triangular_certificate import choose_mesh, validate_mesh  # noqa: E402


class LeakyEnterClient(OfflineMixedSimulatorClient):
    def enter(self) -> dict:
        response = super().enter()
        response["jammer_count"] = 16
        return response


class Q4V3UnknownCountTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Q4Config()
        self.mesh = choose_mesh(self.config)

    def test_route_is_complete_and_mesh_is_strict(self) -> None:
        route = certificate_priority_order((0.0, 0.0), self.mesh, self.config)
        self.assertEqual(set(route), set(range(len(self.mesh.vertices))))
        self.assertTrue(validate_mesh(self.mesh, self.config)["edge_strictly_below_minimum_radius"])

    def test_controller_ignores_leaked_count_and_strictly_terminates(self) -> None:
        sources = [
            OfflineSource(channel, (250.0 + 70.0 * channel, -500.0 + 80.0 * channel), 1300.0, "D" if channel % 2 else "O", 17.0 * channel)
            for channel in range(1, 11)
        ]
        controller = Q4V3Controller(LeakyEnterClient(sources), self.config)
        result = controller.run(self.mesh)
        self.assertFalse(result["runtime_source_count_used"])
        self.assertEqual(result["cleared_count"], 10)
        self.assertEqual(result["absent_count"], 10)
        self.assertTrue(result["strict_termination_verified"])
        self.assertTrue(all(record.status in {ChannelStatus.CLEARED, ChannelStatus.ABSENT} for record in controller.records.values()))


if __name__ == "__main__":
    unittest.main()
