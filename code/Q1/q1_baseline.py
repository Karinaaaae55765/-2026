"""Brute-force diameter baseline for a convex polygon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from q1_geometry import Point, diameter_bruteforce


def baseline_diameter(vertices: Sequence[Point]) -> dict:
    """Return the O(m^2) vertex-pair diameter for comparison."""
    diameter, pair = diameter_bruteforce(vertices)
    return {
        "diameter": diameter,
        "diameter_pair": None if pair is None else [list(pair[0]), list(pair[1])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON file containing vertices_ccw")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    vertices = [tuple(map(float, point)) for point in payload["vertices_ccw"]]
    print(json.dumps(baseline_diameter(vertices), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

