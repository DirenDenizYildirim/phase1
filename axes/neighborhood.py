"""Axis 4f: neighbourhood quality -- an evolvability proxy.

Enumerates every single-voxel type change, keeps the ones that stay inside the published search
space, and measures how well a random subset of those neighbours moves. A morphology surrounded
by capable neighbours is one that a mutation-driven search can improve on.

The feasible fraction is computed over **all** one-voxel changes (9 cells x 4 alternative types =
36 candidates for a 3x3 grid), independently of how many are then simulated.
"""
from __future__ import annotations

import numpy as np

from axes.common import is_feasible
from axes.reach import best_sinusoid
from axes.sim import reduced_sweep_best_dx

AXIS_NAME = "neighborhood"

MAX_NEIGHBORS_SIMULATED = 6   # reduced from the brief's 12 to fit the compute budget; see notes/scope.md
NEIGHBOR_SEED = 777


def all_one_voxel_changes(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    """(feasible neighbours, number of candidate changes considered)."""
    grid = np.asarray(grid, dtype=int)
    rows, cols = grid.shape
    feasible, n_candidates = [], 0
    for r in range(rows):
        for c in range(cols):
            for t in range(5):
                if t == grid[r, c]:
                    continue
                n_candidates += 1
                cand = grid.copy()
                cand[r, c] = t
                if is_feasible(cand):
                    feasible.append(cand)
    return feasible, n_candidates


def compute(grid: np.ndarray) -> dict[str, float]:
    grid = np.asarray(grid, dtype=int)
    neighbors, n_candidates = all_one_voxel_changes(grid)
    feasible_fraction = len(neighbors) / n_candidates if n_candidates else 0.0

    if not neighbors:
        return {
            "nbr_feasible_fraction": float(feasible_fraction),
            "nbr_n_feasible": 0.0,
            "nbr_n_simulated": 0.0,
            "nbr_dx_mean": np.nan, "nbr_dx_max": np.nan, "nbr_dx_std": np.nan,
            "nbr_dx_mean_minus_self": np.nan, "nbr_dx_max_minus_self": np.nan,
        }

    rng = np.random.default_rng(NEIGHBOR_SEED)
    pick = rng.permutation(len(neighbors))[:MAX_NEIGHBORS_SIMULATED]
    dxs = np.array([reduced_sweep_best_dx(neighbors[int(i)]) for i in pick])

    self_dx = float(best_sinusoid(grid)["dx"])
    return {
        "nbr_feasible_fraction": float(feasible_fraction),
        "nbr_n_feasible": float(len(neighbors)),
        "nbr_n_simulated": float(len(dxs)),
        "nbr_dx_mean": float(dxs.mean()),
        "nbr_dx_max": float(dxs.max()),
        "nbr_dx_std": float(dxs.std(ddof=0)),
        # How the neighbourhood compares with the body itself -- is there room to improve?
        "nbr_dx_mean_minus_self": float(dxs.mean() - self_dx),
        "nbr_dx_max_minus_self": float(dxs.max() - self_dx),
    }
