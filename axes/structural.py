"""Axis 4a: structural descriptors. No simulation -- pure functions of the voxel grid."""
from __future__ import annotations

import networkx as nx
import numpy as np

from axes.common import EMPTY, H_ACT, V_ACT

AXIS_NAME = "structural"


def voxel_graph(grid: np.ndarray) -> nx.Graph:
    """4-connectivity adjacency graph over the non-empty voxels."""
    grid = np.asarray(grid)
    g = nx.Graph()
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != EMPTY:
                g.add_node((r, c))
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == EMPTY:
                continue
            for dr, dc in ((0, 1), (1, 0)):
                r2, c2 = r + dr, c + dc
                if r2 < rows and c2 < cols and grid[r2, c2] != EMPTY:
                    g.add_edge((r, c), (r2, c2))
    return g


def _tight_bbox(grid: np.ndarray) -> tuple[int, int]:
    """(height, width) of the tight bounding box around the non-empty voxels."""
    rs, cs = np.nonzero(grid != EMPTY)
    return int(rs.max() - rs.min() + 1), int(cs.max() - cs.min() + 1)


def horizontal_symmetry(grid: np.ndarray) -> float:
    """Fraction of cells that match under a left-right mirror of the tight bounding box.

    1.0 means perfectly left-right symmetric; the minimum is bounded below by the centre
    column, which always matches itself.
    """
    grid = np.asarray(grid)
    rs, cs = np.nonzero(grid != EMPTY)
    sub = grid[rs.min():rs.max() + 1, cs.min():cs.max() + 1]
    return float(np.mean(sub == np.fliplr(sub)))


def modularity(grid: np.ndarray) -> float:
    """Greedy-modularity score of the voxel adjacency graph.

    Returns 0.0 for graphs with no edges, where modularity is undefined.
    """
    g = voxel_graph(grid)
    if g.number_of_edges() == 0:
        return 0.0
    communities = nx.community.greedy_modularity_communities(g)
    return float(nx.community.modularity(g, communities))


def compute(grid: np.ndarray) -> dict[str, float]:
    grid = np.asarray(grid, dtype=int)

    n_vox = int(np.sum(grid != EMPTY))
    n_h = int(np.sum(grid == H_ACT))
    n_v = int(np.sum(grid == V_ACT))
    n_act = n_h + n_v

    bb_h, bb_w = _tight_bbox(grid)
    g = voxel_graph(grid)

    return {
        "n_voxels": float(n_vox),
        "n_actuators": float(n_act),
        # n_act >= 3 and n_vox >= 3 for every feasible grid, so neither ratio divides by zero.
        "actuator_fraction": n_act / n_vox,
        "h_actuator_fraction": n_h / n_act,
        "aspect_ratio": bb_w / bb_h,
        "compactness": n_vox / (bb_h * bb_w),
        "h_symmetry": horizontal_symmetry(grid),
        "modularity": modularity(grid),
        # Constant 1.0 across the feasible set by construction (connectivity is a
        # feasibility rule). Computed anyway so the claim is checked, not assumed;
        # the analysis drops zero-variance columns.
        "n_components": float(nx.number_connected_components(g)),
    }
