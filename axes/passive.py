"""Axis 4b: passive stability. Hold every actuator at neutral and watch the body settle.

Measures whether a morphology is statically stable: does it stay put, sag, topple, or keep
jittering once gravity has had 200 steps to act on it?
"""
from __future__ import annotations

import numpy as np

from axes.sim import run_settle

AXIS_NAME = "passive"


def compute(grid: np.ndarray) -> dict[str, float]:
    """Settle for 200 steps at neutral actuation.

    Returns COM height drift (negative = sagged or collapsed), COM x-drift (movement without
    any actuation, i.e. rolling or toppling), and the largest point-mass speed seen over the
    final 50 steps (near zero = came to rest).
    """
    return run_settle(np.asarray(grid, dtype=int))
