from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

Q2_DIR = Path(__file__).resolve().parents[2] / "code" / "Q2"
sys.path.insert(0, str(Q2_DIR))

from minimum_enclosing_circle import minimum_enclosing_circle
from q2_robust_second_measurement_optimizer import (
    Q2Config,
    build_omega1,
    build_p2_outer_polygon,
    old_two_wedge_status,
    optimize,
)


class MinimumEnclosingCircleTests(unittest.TestCase):
    def test_right_triangle(self) -> None:
        circle = minimum_enclosing_circle([(0, 0), (2, 0), (0, 2)])
        self.assertAlmostEqual(circle.x, 1.0, places=9)
        self.assertAlmostEqual(circle.y, 1.0, places=9)
        self.assertAlmostEqual(circle.radius, math.sqrt(2.0), places=9)

    def test_collinear_points(self) -> None:
        circle = minimum_enclosing_circle([(-2, 0), (0, 0), (3, 0)])
        self.assertAlmostEqual(circle.x, 0.5, places=9)
        self.assertAlmostEqual(circle.radius, 2.5, places=9)


class Q2OptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Q2Config(
            disk_sides=48,
            scenario_pool_count=225,
            candidate_step_m=200.0,
            refinement_step_m=100.0,
            scenario_count=7,
            error_scenario_count=3,
        )

    def test_omega_scenarios_respect_all_physical_constraints(self) -> None:
        omega = build_omega1(self.config)
        points = omega.scenario_pool
        s1_distance = np.linalg.norm(points - np.array([self.config.s1_x, self.config.s1_y]), axis=1)
        arena_distance = np.linalg.norm(points - np.array([self.config.arena_x, self.config.arena_y]), axis=1)
        angle_difference = (
            np.rad2deg(np.arctan2(points[:, 1] - self.config.s1_y, points[:, 0] - self.config.s1_x))
            - self.config.first_bearing_deg
            + 180.0
        ) % 360.0 - 180.0
        self.assertTrue(np.all(s1_distance > self.config.near_radius_m))
        self.assertTrue(np.all(s1_distance <= self.config.maximum_radius_m + 1e-8))
        self.assertTrue(np.all(arena_distance <= self.config.arena_radius_m + 1e-8))
        self.assertTrue(np.all(np.abs(angle_difference) <= self.config.bearing_error_deg + 1e-8))

    def test_cross_zero_first_bearing(self) -> None:
        omega = build_omega1(replace(self.config, first_bearing_deg=359.8))
        self.assertGreaterEqual(len(omega.outer_polygon), 3)
        self.assertGreater(len(omega.scenario_pool), 0)

    def test_canonical_p2_is_bounded_when_old_definition_is_unbounded(self) -> None:
        omega = build_omega1(self.config)
        old_status = old_two_wedge_status(self.config, (-500.0, 0.0), 0.0)
        new_polygon = build_p2_outer_polygon(
            omega.outer_polygon, np.array([-500.0, 0.0]), 0.0, self.config.bearing_error_deg
        )
        self.assertEqual(old_status, "UNBOUNDED")
        self.assertGreaterEqual(len(new_polygon), 3)
        self.assertTrue(np.isfinite(new_polygon).all())

    def test_error_endpoints_keep_true_target_in_p2(self) -> None:
        omega = build_omega1(self.config)
        candidate = np.array([800.0, 600.0])
        target = np.array([1200.0, 0.0])
        true_bearing = math.degrees(math.atan2(target[1] - candidate[1], target[0] - candidate[0]))
        for error in (-self.config.bearing_error_deg, self.config.bearing_error_deg):
            polygon = build_p2_outer_polygon(
                omega.outer_polygon,
                candidate,
                true_bearing + error,
                self.config.bearing_error_deg,
            )
            edges = np.roll(polygon, -1, axis=0) - polygon
            relative = target - polygon
            crosses = edges[:, 0] * relative[:, 1] - edges[:, 1] * relative[:, 0]
            self.assertTrue(np.all(crosses >= -1e-7))

    def test_main_and_baseline_return_certified_candidates(self) -> None:
        for method in ("M2-RR", "M2-ANGLE"):
            result, evidence, _, _ = optimize(self.config, method=method)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertLessEqual(result.certified_max_distance_m, self.config.guaranteed_radius_m + 1e-8)
            self.assertGreater(result.certified_min_distance_m, self.config.near_radius_m)
            self.assertTrue(math.isfinite(result.discretized_worst_radius_m))
            self.assertTrue(evidence["error_endpoints_included"])

    def test_empty_candidate_set_is_reported(self) -> None:
        result, evidence, _, _ = optimize(
            replace(self.config, guaranteed_radius_m=100.0), method="M2-RR"
        )
        self.assertIsNone(result)
        self.assertTrue(evidence["fallback_triggered"])
        self.assertGreater(evidence["minimum_reception_violation_m"], 0.0)

    def test_finer_circle_outer_gap_decreases(self) -> None:
        coarse = build_omega1(self.config)
        fine = build_omega1(replace(self.config, disk_sides=96))
        self.assertLess(fine.circle_outer_gap_m, coarse.circle_outer_gap_m)

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_omega1(replace(self.config, disk_sides=8))
        with self.assertRaises(ValueError):
            build_omega1(replace(self.config, near_radius_m=1600.0))


if __name__ == "__main__":
    unittest.main()
