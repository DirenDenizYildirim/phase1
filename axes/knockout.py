"""Axis 4e: knockout robustness -- how much of the gait survives losing one actuator.

Each of up to 6 randomly chosen actuator voxels is converted to **soft** (type 2, non-actuated)
and the body's best open-loop displacement is recomputed. We report the fraction of the intact
body's displacement that is retained, averaged and at worst.

A knocked-out body may fall outside the published search space (it can drop below the 3-actuator
rule). That is fine -- it is a probe, not a candidate -- but a body with no actuators left cannot
be simulated at all and is skipped.
"""
from __future__ import annotations

import numpy as np

from axes.common import H_ACT, SOFT, V_ACT
from axes.reach import best_sinusoid
from axes.sim import reduced_sweep_best_dx

AXIS_NAME = "knockout"

MAX_KNOCKOUTS = 6
KNOCKOUT_SEED = 4242
# Below this, the intact body barely moves and "fraction retained" is a ratio of two near-zero
# terms, which produces wild outliers (values below -7 were observed at a 0.05 floor). Such rows
# report NaN rather than a meaningless number; the analysis handles missing values explicitly.
MIN_BASE_DX = 0.25


def actuator_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    rs, cs = np.nonzero((grid == H_ACT) | (grid == V_ACT))
    return list(zip(rs.tolist(), cs.tolist()))


def compute(grid: np.ndarray) -> dict[str, float]:
    grid = np.asarray(grid, dtype=int)
    base_dx = float(best_sinusoid(grid)["dx"])

    cells = actuator_cells(grid)
    rng = np.random.default_rng(KNOCKOUT_SEED)
    order = rng.permutation(len(cells))[:MAX_KNOCKOUTS]

    fractions, deltas = [], []
    for i in order:
        r, c = cells[int(i)]
        ko = grid.copy()
        ko[r, c] = SOFT
        if not np.any((ko == H_ACT) | (ko == V_ACT)):
            continue  # no actuators left; EvoGym has nothing to drive
        ko_dx = reduced_sweep_best_dx(ko)
        deltas.append(ko_dx - base_dx)
        if abs(base_dx) >= MIN_BASE_DX:
            fractions.append(ko_dx / base_dx)

    n_tested = len(deltas)
    return {
        "ko_n_tested": float(n_tested),
        "ko_frac_mean": float(np.mean(fractions)) if fractions else np.nan,
        "ko_frac_min": float(np.min(fractions)) if fractions else np.nan,
        "ko_dx_delta_mean": float(np.mean(deltas)) if deltas else np.nan,
        "ko_dx_delta_worst": float(np.min(deltas)) if deltas else np.nan,
    }
