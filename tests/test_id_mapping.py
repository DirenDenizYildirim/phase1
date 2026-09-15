"""Step 2: verify the morphology ID <-> grid mapping and the feasibility rules.

The ground truth's search space is 3x3 (see notes/dataset.md): IDs run over 0..5**9-1 and the
1,305,840 feasible ones are exactly the keys of the published results dicts.

These tests pin down the mapping with hard-coded examples so they keep running in CI without the
3 GB upstream data. When the upstream clone *is* present, two extra tests cross-check our
re-implementation against `search_space.py` itself and against the published key set.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from axes.common import H_ACT, V_ACT, grid_to_idx, idx_to_grid, is_feasible

BB = (3, 3)
REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "data" / "mc-landscape"
GROUND_TRUTH = REPO_ROOT / "data" / "ground_truth.parquet"

# Five morphologies spanning the published ID range, taken from the dataset's own key list.
# Every one of these is a key in updated_results.pkl, so every one must be feasible.
KNOWN = {
    93:      [0, 0, 0, 0, 0, 0, 3, 3, 3],   # smallest feasible ID: a row of 3 h-actuators
    601956:  [1, 2, 3, 2, 3, 0, 3, 1, 1],
    1108695: [2, 4, 0, 4, 3, 4, 2, 4, 0],
    1535356: [3, 4, 3, 1, 1, 2, 4, 1, 1],
    1953124: [4, 4, 4, 4, 4, 4, 4, 4, 4],   # largest ID, 5**9 - 1: all vertical actuators
}


@pytest.mark.parametrize("idx,flat", KNOWN.items())
def test_known_ids_decode_to_expected_grids(idx, flat):
    grid = idx_to_grid(idx, BB)
    assert grid.shape == BB
    assert grid.flatten().tolist() == flat


@pytest.mark.parametrize("idx", KNOWN)
def test_five_reconstructed_morphologies_are_feasible(idx):
    """Step 2's explicit check: reconstruct 5 morphologies, confirm the repo's rules pass."""
    grid = idx_to_grid(idx, BB)
    assert is_feasible(grid), f"id {idx} is a published key but failed the feasibility rules"


@pytest.mark.parametrize("idx", KNOWN)
def test_roundtrip_id_to_grid_to_id(idx):
    assert grid_to_idx(idx_to_grid(idx, BB)) == idx


def test_id_is_base_five_row_major():
    """The ID is literally the base-5 numeral formed by the voxel types in row-major order."""
    grid = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    assert grid_to_idx(grid) == 1
    grid = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    assert grid_to_idx(grid) == 5
    grid = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert grid_to_idx(grid) == 5 ** 8


def test_infeasible_examples_are_rejected():
    # all empty -> not connected, no voxels, no actuators
    assert not is_feasible(np.zeros(BB, dtype=int))
    # a single voxel -> fewer than 3 voxels
    g = np.zeros(BB, dtype=int); g[2, 2] = H_ACT
    assert not is_feasible(g)
    # 3 connected voxels but only 2 actuators
    g = np.zeros(BB, dtype=int); g[2, :] = [1, H_ACT, V_ACT]
    assert not is_feasible(g)
    # 3 actuators but disconnected (opposite corners)
    g = np.zeros(BB, dtype=int)
    g[0, 0] = H_ACT; g[0, 1] = H_ACT; g[2, 2] = V_ACT
    assert not is_feasible(g)


def test_actuator_rule_counts_both_orientations():
    g = np.zeros(BB, dtype=int)
    g[2, :] = [H_ACT, V_ACT, H_ACT]
    assert is_feasible(g)
    g[2, :] = [H_ACT, V_ACT, 1]          # only 2 actuators
    assert not is_feasible(g)


# --- cross-checks that need the upstream clone / derived ground truth ---

def _load_upstream():
    spec = importlib.util.spec_from_file_location("upstream_search_space",
                                                  UPSTREAM / "search_space.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not (UPSTREAM / "search_space.py").exists(),
                    reason="upstream clone not present (data/mc-landscape is gitignored)")
def test_matches_upstream_search_space_implementation():
    """Our mapping must agree with upstream `search_space.py` digit for digit."""
    up = _load_upstream()
    rng = np.random.default_rng(12345)
    for idx in [0, 1, 93, 1953124] + rng.integers(0, 5 ** 9, size=200).tolist():
        idx = int(idx)
        theirs = up.integer_idx_to_ndarray(idx, BB)
        ours = idx_to_grid(idx, BB)
        assert np.array_equal(theirs, ours), idx
        assert up.ndarray_to_integer_idx(ours) == grid_to_idx(ours) == idx


@pytest.mark.skipif(not GROUND_TRUTH.exists(), reason="data/ground_truth.parquet not built")
def test_published_keys_are_exactly_the_feasible_set_on_a_sample():
    """Published IDs must be feasible; IDs absent from the map must be infeasible."""
    import pandas as pd

    published = set(pd.read_parquet(GROUND_TRUTH)["id"].tolist())
    rng = np.random.default_rng(999)
    sample = rng.integers(0, 5 ** 9, size=500).tolist()
    for idx in sample:
        idx = int(idx)
        assert is_feasible(idx_to_grid(idx, BB)) == (idx in published), idx
