"""Shared helpers: morphology ID <-> grid mapping, feasibility, and the probe environment.

The ID <-> grid mapping and the feasibility rules are deliberate re-implementations of
``search_space.py`` from ``mertan-a/morphology-fitness-landscape`` so that this project does not
depend on importing that repo at runtime. ``tests/test_id_mapping.py`` checks the two agree.
"""
from __future__ import annotations

import numpy as np

# EvoGym voxel type codes.
EMPTY, RIGID, SOFT, H_ACT, V_ACT = 0, 1, 2, 3, 4
VOXEL_NAMES = {EMPTY: "empty", RIGID: "rigid", SOFT: "soft",
               H_ACT: "h_actuator", V_ACT: "v_actuator"}

# The ground truth's search space is 3x3, NOT 5x5 -- verified exactly: our independent
# enumeration of feasible 3x3 grids reproduces the published key set (1,305,840 IDs,
# min 93, max 5**9-1) as an exact set match. See notes/dataset.md.
BOUNDING_BOX = (3, 3)

# --- Environment config, matched to the ground truth (see notes/dataset.md) ---
TASK = "Walker-v0"
MAX_EPISODE_STEPS = 500
# Actuator action range in EvoGym: target volume ratio. 1.0 is neutral (no deformation).
ACTION_LOW, ACTION_HIGH, ACTION_NEUTRAL = 0.6, 1.6, 1.0


def idx_to_grid(idx: int, bounding_box: tuple[int, int] = BOUNDING_BOX) -> np.ndarray:
    """Integer morphology ID -> voxel grid.

    Mirrors ``search_space.integer_idx_to_ndarray``: the ID is read as a base-5 numeral whose
    digits, left-padded with zeros to ``rows * cols``, are the voxel types in row-major order.
    """
    rows, cols = bounding_box
    n = rows * cols
    s = np.base_repr(int(idx), base=5)
    if len(s) < n:
        s = "0" * (n - len(s)) + s
    if len(s) > n:
        raise ValueError(f"id {idx} does not fit in a {rows}x{cols} grid")
    return np.array([int(c) for c in s], dtype=int).reshape(rows, cols)


def grid_to_idx(grid: np.ndarray) -> int:
    """Voxel grid -> integer morphology ID. Mirrors ``search_space.ndarray_to_integer_idx``."""
    return int("".join(np.asarray(grid).flatten().astype(int).astype(str)), 5)


def is_feasible(grid: np.ndarray) -> bool:
    """The upstream search-space membership test, all three rules.

    1. the filled voxels form one connected component (EvoGym's ``is_connected``);
    2. at least 3 non-empty voxels;
    3. at least 3 actuator voxels (horizontal + vertical).
    """
    from evogym import is_connected

    grid = np.asarray(grid, dtype=int)
    if not is_connected(grid):
        return False
    if np.sum(grid > 0) < 3:
        return False
    if np.sum(grid == H_ACT) + np.sum(grid == V_ACT) < 3:
        return False
    return True


def n_actuators(grid: np.ndarray) -> int:
    grid = np.asarray(grid)
    return int(np.sum(grid == H_ACT) + np.sum(grid == V_ACT))


def make_env(grid: np.ndarray, seed: int | None = None):
    """Build the probe environment for a morphology.

    Matches the ground truth's environment: ``Walker-v0`` (flat-ground locomotion) with full
    voxel connectivity and a 500-step episode cap. We do NOT apply the ground truth's
    ``RewardShapingWrapper`` because none of our axes read the reward signal -- they read
    centre-of-mass kinematics straight off the simulator.
    """
    import gymnasium as gym
    from evogym import get_full_connectivity
    import evogym.envs  # noqa: F401  (importing registers the EvoGym env ids)

    grid = np.asarray(grid, dtype=int)
    env = gym.make(TASK, body=grid, connections=get_full_connectivity(grid))
    env.reset(seed=seed)
    if seed is not None:
        env.action_space.seed(seed)
    return env


def robot_com(env) -> np.ndarray:
    """Centre of mass (x, y) of the robot, as a length-2 array, from the live simulator."""
    uw = env.unwrapped
    pos = uw.object_pos_at_time(uw.get_time(), "robot")  # (2, n_point_masses)
    return pos.mean(axis=1)


def robot_vel(env) -> np.ndarray:
    """Per-point-mass velocities of the robot, shape (2, n_point_masses)."""
    uw = env.unwrapped
    return uw.object_vel_at_time(uw.get_time(), "robot")
