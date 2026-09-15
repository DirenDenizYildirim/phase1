"""Axis 4c: reach -- how far the body travels under untuned open-loop actuation.

Two probes, neither of which tunes anything:

* **random**: 8 episodes of uniform random actions over the actuator range;
* **sinusoidal**: a 4 frequency x 4 phase-pattern sweep, taking the best net displacement.
  This is the "best open-loop gait" measure.
"""
from __future__ import annotations

import numpy as np

from axes.sim import (FREQUENCIES, PHASE_PATTERNS, best_of, run_random_episodes,
                      sinusoid_sweep)

AXIS_NAME = "reach"


def best_sinusoid(grid: np.ndarray) -> dict:
    """The best entry of the (memoised) sinusoid sweep, by net forward displacement."""
    return best_of(sinusoid_sweep(np.asarray(grid, dtype=int)))


def compute(grid: np.ndarray) -> dict[str, float]:
    grid = np.asarray(grid, dtype=int)

    eps = run_random_episodes(grid)
    dx = np.array([e["dx"] for e in eps])
    speeds = np.array([e["mean_speed"] for e in eps])

    sweep = sinusoid_sweep(grid)
    best = best_of(sweep)
    all_dx = np.array([r["dx"] for r in sweep])

    return {
        # --- random actuation ---
        "rand_dx_mean": float(dx.mean()),
        "rand_dx_max": float(dx.max()),
        "rand_dx_std": float(dx.std(ddof=0)),
        "rand_mean_speed": float(speeds.mean()),
        # --- sinusoidal actuation: the best open-loop gait ---
        "sin_best_dx": float(best["dx"]),
        # Same gait scored over the ground truth's own 100-step horizon.
        "sin_best_dx_short": float(best["dx_short"]),
        "sin_best_freq": float(best["freq"]),
        "sin_best_pattern_id": float(PHASE_PATTERNS.index(best["pattern"])),
        # Spread across the sweep: how much the gait choice matters for this body.
        "sin_dx_mean": float(all_dx.mean()),
        "sin_dx_std": float(all_dx.std(ddof=0)),
        "sin_dx_range": float(all_dx.max() - all_dx.min()),
    }
