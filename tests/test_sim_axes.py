"""Unit tests for the simulation-backed axes (4b-4f).

Each test uses one small morphology and asserts shape, finiteness, determinism and a handful of
properties that must hold by construction. They are deliberately cheap -- the point is that the
plumbing is right, not that the physics is re-derived.
"""
import numpy as np
import pytest

from axes import efficiency, knockout, neighborhood, passive, reach
from axes.common import H_ACT, SOFT, V_ACT, idx_to_grid, is_feasible
from axes.sim import (PHASE_PATTERNS, REDUCED_SWEEP, actuator_rowcol, phase_offsets,
                      sinusoid_actions, ACTION_HIGH, ACTION_LOW)

BB = (3, 3)
# A biped-ish body with actuators in known places; feasible, and cheap to simulate.
GRID = idx_to_grid(1514953, BB)   # [[3,4,1],[4,3,4],[3,0,3]]


def test_fixture_is_feasible():
    assert is_feasible(GRID)


# --- action construction ---

def test_actuator_rowcol_is_row_major_action_order():
    rc = actuator_rowcol(GRID)
    flat = GRID.flatten()
    expected = np.flatnonzero(flat >= H_ACT)
    assert len(rc) == len(expected)
    assert [r * 3 + c for r, c in rc] == expected.tolist()


def test_phase_patterns_split_the_body_as_named():
    rc = actuator_rowcol(GRID)
    lr = phase_offsets(GRID, "left_right")
    # column 0 and 1 are at/left of centre -> phase 0; column 2 -> phase pi
    assert np.all(lr[rc[:, 1] <= 1] == 0.0)
    assert np.all(lr[rc[:, 1] == 2] == pytest.approx(np.pi))

    tb = phase_offsets(GRID, "top_bottom")
    assert np.all(tb[rc[:, 0] <= 1] == 0.0)
    assert np.all(tb[rc[:, 0] == 2] == pytest.approx(np.pi))

    assert np.all(phase_offsets(GRID, "in_phase") == 0.0)


def test_random_phase_pattern_is_deterministic():
    a = phase_offsets(GRID, "random")
    b = phase_offsets(GRID, "random")
    assert np.array_equal(a, b)


def test_sinusoid_actions_stay_inside_the_action_space():
    phases = phase_offsets(GRID, "random")
    for t in range(0, 120):
        a = sinusoid_actions(t, 0.1, phases)
        assert np.all(a >= ACTION_LOW - 1e-12) and np.all(a <= ACTION_HIGH + 1e-12)


def test_unknown_phase_pattern_raises():
    with pytest.raises(ValueError):
        phase_offsets(GRID, "diagonal")


# --- 4b passive ---

def test_passive_returns_finite_settle_statistics():
    f = passive.compute(GRID)
    assert set(f) == {"settle_com_y_drift", "settle_com_x_drift",
                      "settle_max_speed_tail", "settle_mean_speed_tail"}
    assert all(np.isfinite(v) for v in f.values())
    # Held at neutral on flat ground, a body should not wander far sideways.
    assert abs(f["settle_com_x_drift"]) < 1.0
    assert f["settle_max_speed_tail"] >= f["settle_mean_speed_tail"] >= 0.0


# --- 4c reach ---

def test_reach_keys_and_internal_consistency():
    f = reach.compute(GRID)
    assert f["rand_dx_max"] >= f["rand_dx_mean"]
    assert f["rand_dx_std"] >= 0.0
    assert f["rand_mean_speed"] >= 0.0
    # the best of the sweep is at least its mean, and the range is non-negative
    assert f["sin_best_dx"] >= f["sin_dx_mean"]
    assert f["sin_dx_range"] >= 0.0
    assert 0 <= f["sin_best_pattern_id"] < len(PHASE_PATTERNS)
    assert all(np.isfinite(v) for v in f.values())


def test_reach_is_deterministic():
    assert reach.compute(GRID) == reach.compute(GRID)


def test_best_sinusoid_beats_naive_in_phase_actuation_on_this_body():
    """Sanity: phase-offset gaits should out-travel the degenerate all-together gait."""
    from axes.sim import run_sinusoid
    in_phase = run_sinusoid(GRID, 0.1, "in_phase")["dx"]
    assert reach.best_sinusoid(GRID)["dx"] > in_phase


# --- 4d efficiency ---

def test_efficiency_denominators_and_signs():
    f = efficiency.compute(GRID)
    assert f["eff_action_abs_sum"] > 0
    assert f["eff_action_dev_sum"] > 0
    # |action| >= |action - 1.0| pointwise over [0.6, 1.6], so the raw denominator is larger
    assert f["eff_action_abs_sum"] > f["eff_action_dev_sum"]
    dx = reach.best_sinusoid(GRID)["dx"]
    assert np.sign(f["eff_raw"]) == np.sign(dx)
    # the log compression preserves sign and shrinks magnitude
    assert np.sign(f["eff_raw_log"]) == np.sign(f["eff_raw"])
    assert abs(f["eff_raw_log"]) <= abs(f["eff_raw"]) + 1e-12


def test_signed_log1p_handles_negatives():
    assert efficiency.signed_log1p(0.0) == 0.0
    assert efficiency.signed_log1p(-2.0) == pytest.approx(-np.log1p(2.0))


# --- 4e knockout ---

def test_knockout_converts_actuators_to_soft_and_reports_fractions():
    f = knockout.compute(GRID)
    n_act = int(np.sum((GRID == H_ACT) | (GRID == V_ACT)))
    assert f["ko_n_tested"] == min(knockout.MAX_KNOCKOUTS, n_act)
    assert f["ko_frac_min"] <= f["ko_frac_mean"]
    assert f["ko_dx_delta_worst"] <= f["ko_dx_delta_mean"]


def test_knockout_cells_are_only_actuators():
    cells = knockout.actuator_cells(GRID)
    assert cells
    for r, c in cells:
        assert GRID[r, c] in (H_ACT, V_ACT)


def test_knockout_of_a_three_actuator_body_is_still_simulable():
    """Knocking out an actuator can leave the search space; it must still be measurable."""
    grid = np.array([[0, 0, 0], [1, 1, 0], [3, 3, 3]])
    assert is_feasible(grid)
    ko = grid.copy(); ko[2, 0] = SOFT
    assert not is_feasible(ko)          # only 2 actuators left -> out of the search space
    f = knockout.compute(grid)
    assert f["ko_n_tested"] == 3
    assert np.isfinite(f["ko_dx_delta_mean"])


# --- 4f neighborhood ---

def test_neighborhood_candidate_count_is_every_single_voxel_change():
    _, n_candidates = neighborhood.all_one_voxel_changes(GRID)
    assert n_candidates == 9 * 4      # 9 cells x 4 alternative types


def test_neighborhood_neighbours_differ_in_exactly_one_cell_and_are_feasible():
    neighbors, _ = neighborhood.all_one_voxel_changes(GRID)
    assert neighbors
    for nb in neighbors:
        assert int(np.sum(nb != GRID)) == 1
        assert is_feasible(nb)


def test_neighborhood_outputs():
    f = neighborhood.compute(GRID)
    assert 0.0 <= f["nbr_feasible_fraction"] <= 1.0
    assert f["nbr_n_simulated"] == min(neighborhood.MAX_NEIGHBORS_SIMULATED, f["nbr_n_feasible"])
    assert f["nbr_dx_max"] >= f["nbr_dx_mean"]
    assert f["nbr_dx_std"] >= 0.0


def test_reduced_sweep_is_the_documented_three_patterns():
    assert REDUCED_SWEEP == ((0.10, "left_right"), (0.10, "top_bottom"), (0.10, "random"))
