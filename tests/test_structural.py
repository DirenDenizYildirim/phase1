"""Unit tests for axis 4a (structural)."""
import numpy as np
import pytest

from axes.common import idx_to_grid, is_feasible
from axes.structural import compute, horizontal_symmetry, modularity, voxel_graph

BB = (3, 3)


def test_counts_and_fractions_on_a_hand_built_grid():
    #  3 4 1      3 non-empty in row 0, etc.
    #  4 3 0
    #  3 0 0
    grid = np.array([[3, 4, 1], [4, 3, 0], [3, 0, 0]])
    f = compute(grid)
    assert f["n_voxels"] == 6
    assert f["n_actuators"] == 5          # three 3s and two 4s
    assert f["actuator_fraction"] == pytest.approx(5 / 6)
    assert f["h_actuator_fraction"] == pytest.approx(3 / 5)


def test_bounding_box_is_tight_not_the_full_grid():
    # a 1x3 horizontal bar in the bottom row
    grid = np.array([[0, 0, 0], [0, 0, 0], [3, 3, 3]])
    f = compute(grid)
    assert f["aspect_ratio"] == pytest.approx(3.0)   # 3 wide, 1 tall
    assert f["compactness"] == pytest.approx(1.0)    # fills its bbox

    # a 3x1 vertical bar
    grid = np.array([[4, 0, 0], [4, 0, 0], [4, 0, 0]])
    f = compute(grid)
    assert f["aspect_ratio"] == pytest.approx(1 / 3)
    assert f["compactness"] == pytest.approx(1.0)


def test_compactness_below_one_when_bbox_has_holes():
    grid = np.array([[3, 0, 3], [0, 0, 0], [3, 0, 3]])
    f = compute(grid)
    assert f["compactness"] == pytest.approx(4 / 9)


def test_horizontal_symmetry():
    sym = np.array([[3, 1, 3], [4, 2, 4], [3, 0, 3]])
    assert horizontal_symmetry(sym) == pytest.approx(1.0)

    asym = np.array([[3, 3, 3], [1, 0, 0], [1, 0, 0]])
    # tight bbox is the whole grid; mirrored, row0 matches (3/3), rows 1-2 match only
    # in the centre column (1/3 each) -> 5/9
    assert horizontal_symmetry(asym) == pytest.approx(5 / 9)


def test_modularity_is_zero_without_edges_and_positive_for_a_dumbbell():
    # Feasibility guarantees connectivity, but the function must not blow up regardless.
    single = np.array([[3, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert modularity(single) == 0.0

    # two dense blobs joined by one edge -> a clear community split
    grid = np.array([[3, 3, 0], [3, 1, 0], [0, 4, 4]])
    assert modularity(grid) > 0.0


def test_voxel_graph_uses_4_connectivity_not_8():
    # two voxels touching only at a corner must NOT be joined
    grid = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 0]])
    g = voxel_graph(grid)
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 0


def test_all_outputs_are_finite_floats_over_real_morphologies():
    for idx in (93, 601956, 1108695, 1535356, 1953124, 1514953):
        grid = idx_to_grid(idx, BB)
        assert is_feasible(grid)
        f = compute(grid)
        assert set(f) == {"n_voxels", "n_actuators", "actuator_fraction", "h_actuator_fraction",
                          "aspect_ratio", "compactness", "h_symmetry", "modularity",
                          "n_components"}
        for k, v in f.items():
            assert isinstance(v, float), k
            assert np.isfinite(v), k
        # connectivity is a feasibility rule, so this must hold for every feasible grid
        assert f["n_components"] == 1.0
