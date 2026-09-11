"""Q2批准使用的M2-ANGLE交会角基线。"""

from __future__ import annotations

from q2_robust_second_measurement_optimizer import CandidateResult, Q2Config, optimize


def run_angle_baseline(config: Q2Config) -> tuple[CandidateResult | None, dict, object, object]:
    return optimize(config, method="M2-ANGLE")
