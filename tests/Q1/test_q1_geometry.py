from __future__ import annotations

import math
import itertools
import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[2] / "code" / "Q1"
sys.path.insert(0, str(CODE_DIR))

from q1_geometry import (  # noqa: E402
    Observation,
    convex_hull,
    diameter_bruteforce,
    diameter_circle_report,
    diameter_rotating_calipers,
    localize_source,
    point_in_all_wedges,
)


def bearing(source: tuple[float, float], detector: tuple[float, float]) -> float:
    return math.degrees(math.atan2(source[1] - detector[1], source[0] - detector[0])) % 360.0


class Q1GeometryTests(unittest.TestCase):
    def test_bounded_region_contains_true_source(self) -> None:
        source = (300.0, 200.0)
        detectors = [(-600.0, -200.0), (900.0, -400.0), (700.0, 900.0), (-500.0, 700.0)]
        observations = [Observation(*point, bearing(source, point)) for point in detectors]
        result = localize_source(observations)
        self.assertEqual(result["status"], "OK")
        self.assertTrue(point_in_all_wedges(source, observations, tol=1e-7))
        self.assertAlmostEqual(
            result["diameter"], result["diagnostics"]["diameter_bruteforce"], places=9
        )

    def test_cross_zero_bearings(self) -> None:
        source = (1000.0, 0.0)
        detectors = [(0.0, 10.0), (0.0, -10.0), (1600.0, 400.0), (1600.0, -400.0)]
        observations = [Observation(*point, bearing(source, point)) for point in detectors]
        result = localize_source(observations)
        self.assertEqual(result["status"], "OK")
        self.assertTrue(point_in_all_wedges(source, observations, tol=1e-7))

    def test_error_endpoints_are_inclusive(self) -> None:
        source = (500.0, 250.0)
        detectors = [(-400.0, -300.0), (900.0, -500.0), (1000.0, 900.0), (-300.0, 800.0)]
        observations = []
        for index, detector in enumerate(detectors):
            endpoint_error = 1.0 if index % 2 == 0 else -1.0
            observations.append(Observation(*detector, bearing(source, detector) + endpoint_error))
        result = localize_source(observations)
        self.assertIn(result["status"], {"OK", "DEGENERATE"})
        self.assertTrue(point_in_all_wedges(source, observations, tol=1e-7))

    def test_unbounded_single_wedge(self) -> None:
        result = localize_source([Observation(0.0, 0.0, 45.0)])
        self.assertEqual(result["status"], "UNBOUNDED")
        self.assertIsNone(result["diameter"])
        self.assertIsNotNone(result["diagnostics"]["recession_direction"])

    def test_very_narrow_wedge_remains_unbounded(self) -> None:
        result = localize_source(
            [Observation(1_000_000.0, -1_000_000.0, 359.999999)],
            delta_deg=1e-7,
        )
        self.assertEqual(result["status"], "UNBOUNDED")

    def test_empty_intersection(self) -> None:
        result = localize_source(
            [Observation(0.0, 0.0, 0.0), Observation(0.0, 10.0, 180.0)]
        )
        self.assertEqual(result["status"], "EMPTY")

    def test_degenerate_shared_apex(self) -> None:
        result = localize_source(
            [Observation(0.0, 0.0, 0.0), Observation(0.0, 0.0, 180.0)]
        )
        self.assertEqual(result["status"], "DEGENERATE")
        self.assertAlmostEqual(result["diameter"], 0.0)

    def test_equilateral_triangle_not_covered(self) -> None:
        triangle = [(0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3.0) / 2.0)]
        diameter, pair = diameter_rotating_calipers(triangle)
        self.assertIsNotNone(pair)
        report = diameter_circle_report(triangle, pair, 1e-12)
        self.assertAlmostEqual(diameter, 1.0)
        self.assertFalse(report["covers_region"])

    def test_random_main_matches_baseline(self) -> None:
        import random

        generator = random.Random(2026)
        for _ in range(50):
            points = [(generator.gauss(0, 1), generator.gauss(0, 1)) for _ in range(30)]
            hull = convex_hull(points)
            main, _ = diameter_rotating_calipers(hull, 1e-12)
            baseline, _ = diameter_bruteforce(hull)
            self.assertAlmostEqual(main, baseline, places=10)

    def test_observation_order_does_not_change_bounded_result(self) -> None:
        source = (300.0, 200.0)
        detectors = [(-600.0, -200.0), (900.0, -400.0), (700.0, 900.0), (-500.0, 700.0)]
        observations = [Observation(*point, bearing(source, point)) for point in detectors]
        reference = localize_source(observations)
        for order in itertools.permutations(observations):
            candidate = localize_source(order)
            self.assertEqual(candidate["status"], reference["status"])
            self.assertAlmostEqual(candidate["diameter"], reference["diameter"], places=8)
            self.assertEqual(candidate["diameter_circle"]["covers_region"], reference["diameter_circle"]["covers_region"])

    def test_constraint_diagnostics_are_within_tolerance(self) -> None:
        source = (300.0, 200.0)
        detectors = [(-600.0, -200.0), (900.0, -400.0), (700.0, 900.0), (-500.0, 700.0)]
        result = localize_source([Observation(*point, bearing(source, point)) for point in detectors])
        diagnostics = result["diagnostics"]
        self.assertLessEqual(diagnostics["maximum_constraint_violation"], diagnostics["tolerance"])

    def test_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            localize_source([])
        with self.assertRaises(ValueError):
            localize_source([Observation(0.0, 0.0, 0.0)], delta_deg=90.0)
        with self.assertRaises(ValueError):
            localize_source([{"x": 0.0, "y": 0.0}])
        with self.assertRaises(ValueError):
            point_in_all_wedges((0.0, 0.0), [Observation(0.0, 0.0, 0.0)], delta_deg=0.0)
        with self.assertRaises(ValueError):
            point_in_all_wedges((math.inf, 0.0), [Observation(0.0, 0.0, 0.0)])
        with self.assertRaises(ValueError):
            point_in_all_wedges((0.0, 0.0), [Observation(0.0, 0.0, 0.0)], tol=-1.0)


if __name__ == "__main__":
    unittest.main()
