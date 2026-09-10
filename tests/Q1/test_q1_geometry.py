from __future__ import annotations

import math
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

    def test_unbounded_single_wedge(self) -> None:
        result = localize_source([Observation(0.0, 0.0, 45.0)])
        self.assertEqual(result["status"], "UNBOUNDED")
        self.assertIsNone(result["diameter"])

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

    def test_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            localize_source([])
        with self.assertRaises(ValueError):
            localize_source([Observation(0.0, 0.0, 0.0)], delta_deg=90.0)
        with self.assertRaises(ValueError):
            localize_source([{"x": 0.0, "y": 0.0}])


if __name__ == "__main__":
    unittest.main()

