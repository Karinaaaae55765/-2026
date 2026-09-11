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
    build_candidate_region_samples,
    build_omega1,
    build_p2_outer_polygon,
    candidate_metrics,
    classify_candidates,
    evaluate_candidate,
    old_two_wedge_status,
    optimize,
)


class MinimumEnclosingCircleTests(unittest.TestCase):
    def test_triangle_and_collinear_cases(self) -> None:
        triangle = minimum_enclosing_circle([(0, 0), (2, 0), (0, 2)])
        self.assertAlmostEqual(triangle.radius, math.sqrt(2.0), places=9)
        line = minimum_enclosing_circle([(-2, 0), (0, 0), (3, 0)])
        self.assertAlmostEqual(line.x, 0.5, places=9)
        self.assertAlmostEqual(line.radius, 2.5, places=9)


class Q2ConditionalReceptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Q2Config(
            disk_sides=48,
            scenario_pool_count=225,
            candidate_step_m=200.0,
            candidate_region_step_m=100.0,
            refinement_step_m=100.0,
            scenario_count=7,
            error_scenario_count=3,
        )

    def test_omega_respects_first_success(self) -> None:
        omega = build_omega1(self.config)
        points = omega.scenario_pool
        distance = np.linalg.norm(points, axis=1)
        angle = (np.rad2deg(np.arctan2(points[:, 1], points[:, 0])) + 180.0) % 360.0 - 180.0
        self.assertTrue(np.all(distance > 5.0))
        self.assertTrue(np.all(distance <= 1500.0 + 1e-8))
        self.assertTrue(np.all(np.abs(angle) <= 1.0 + 1e-8))

    def test_conditional_radius_recovers_point_rejected_by_old_rule(self) -> None:
        omega = build_omega1(self.config)
        point = np.array([[500.0, 800.0]])
        classified = classify_candidates(point, omega.scenario_pool, self.config)
        _, _, maximum, _ = candidate_metrics(point, omega.scenario_pool, self.config)
        self.assertEqual(len(classified["signal"]), 1)
        self.assertGreater(maximum[0], self.config.guaranteed_radius_m)

    def test_same_location_is_not_a_second_measurement_point(self) -> None:
        omega = build_omega1(self.config)
        classified = classify_candidates(np.array([[0.0, 0.0]]), omega.scenario_pool, self.config)
        self.assertEqual(len(classified["signal"]), 0)

    def test_near_is_a_success_branch(self) -> None:
        omega = build_omega1(self.config)
        candidate = np.array([750.0, 0.0])
        evaluation = evaluate_candidate(
            candidate,
            omega.outer_polygon,
            np.array([[750.0, 0.0], [1200.0, 0.0]]),
            [-1.0, 1.0],
            self.config,
        )
        self.assertEqual(evaluation.near_scenario_count, 1)
        self.assertEqual(evaluation.bearing_scenario_count, 2)

    def test_signal_and_bearing_regions_are_reported_separately(self) -> None:
        omega = build_omega1(self.config)
        region = build_candidate_region_samples(omega, self.config)
        self.assertGreater(len(region["signal"]), 0)
        self.assertGreater(len(region["bearing"]), 0)
        self.assertLessEqual(len(region["bearing"]), len(region["signal"]))

    def test_canonical_p2_is_bounded_and_contains_endpoint_target(self) -> None:
        omega = build_omega1(self.config)
        candidate = np.array([500.0, 0.0])
        self.assertEqual(old_two_wedge_status(self.config, tuple(candidate), 0.0), "UNBOUNDED")
        target = np.array([1200.0, 0.0])
        true_bearing = math.degrees(math.atan2(target[1] - candidate[1], target[0] - candidate[0]))
        for error in (-1.0, 1.0):
            polygon = build_p2_outer_polygon(
                omega.outer_polygon,
                candidate,
                true_bearing + error,
                self.config.bearing_error_deg,
                self.config.maximum_radius_m,
                self.config.disk_sides,
            )
            self.assertGreaterEqual(len(polygon), 3)
            edges = np.roll(polygon, -1, axis=0) - polygon
            relative = target - polygon
            cross = edges[:, 0] * relative[:, 1] - edges[:, 1] * relative[:, 0]
            self.assertTrue(np.all(cross >= -1e-7) or np.all(cross <= 1e-7))

    def test_cross_zero_and_finer_circle_resolution(self) -> None:
        coarse = build_omega1(replace(self.config, first_bearing_deg=359.8))
        fine = build_omega1(replace(self.config, disk_sides=96))
        self.assertGreaterEqual(len(coarse.outer_polygon), 3)
        self.assertLess(fine.circle_outer_gap_m, build_omega1(self.config).circle_outer_gap_m)

    def test_main_and_baseline_use_same_signal_region(self) -> None:
        for method in ("M2-RR", "M2-ANGLE"):
            result, evidence, _, candidates = optimize(self.config, method)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertGreaterEqual(result.sampled_signal_margin_m, -self.config.tie_tolerance_m)
            self.assertGreater(len(candidates), 0)
            self.assertTrue(evidence["conditional_reception_model"])
            self.assertTrue(math.isfinite(result.discretized_worst_radius_m))

    def test_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            build_omega1(replace(self.config, disk_sides=8))
        with self.assertRaises(ValueError):
            build_omega1(replace(self.config, near_radius_m=1600.0))


if __name__ == "__main__":
    unittest.main()
