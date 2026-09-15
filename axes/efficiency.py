"""Axis 4d: efficiency of the best open-loop gait -- distance per unit of actuation.

Two denominators are reported:

* `eff_raw` uses the literal sum of |action| the brief asks for. EvoGym actions are volume
  ratios in [0.6, 1.6] and are never near zero, so this denominator is dominated by
  `n_actuators x steps` and the measure is close to "displacement per actuator".
* `eff_dev` divides instead by the summed deviation from neutral, |action - 1.0|, which is the
  part of the signal that actually deforms a voxel and so is the more physical notion of
  actuation cost.

Both are reported raw and log-compressed. Displacement can be negative, so the compression is
the sign-preserving `sign(x) * log1p(|x|)` rather than `log1p(x)`, which is undefined below -1.
"""
from __future__ import annotations

import numpy as np

from axes.reach import best_sinusoid

AXIS_NAME = "efficiency"


def signed_log1p(x: float) -> float:
    return float(np.sign(x) * np.log1p(abs(x)))


def compute(grid: np.ndarray) -> dict[str, float]:
    best = best_sinusoid(np.asarray(grid, dtype=int))
    dx = float(best["dx"])
    abs_sum = float(best["action_abs_sum"])
    dev_sum = float(best["action_dev_sum"])

    # abs_sum is bounded below by 0.6 * n_actuators * steps > 0, so it never divides by zero.
    eff_raw = dx / abs_sum
    # dev_sum is zero only if the gait never leaves neutral, which no sinusoid does.
    eff_dev = dx / dev_sum if dev_sum > 0 else 0.0

    return {
        "eff_raw": eff_raw,
        "eff_raw_log": signed_log1p(eff_raw),
        "eff_dev": eff_dev,
        "eff_dev_log": signed_log1p(eff_dev),
        "eff_action_abs_sum": abs_sum,
        "eff_action_dev_sum": dev_sum,
    }
