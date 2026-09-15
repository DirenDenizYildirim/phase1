"""Step 1 smoke test: EvoGym is installed and a random valid body simulates.

Run for both the project's working 3x3 space and the 5x5 space named in the original spec.
"""
import numpy as np
import pytest

from axes.common import (ACTION_HIGH, ACTION_LOW, BOUNDING_BOX, MAX_EPISODE_STEPS,
                         TASK, is_feasible, make_env, robot_com)

SMOKE_SEED = 20240517


GRID_SIZES = [(3, 3), (5, 5)]


def random_valid_grid(rng, size=BOUNDING_BOX, max_tries=10000):
    """Rejection-sample a feasible grid of the given size."""
    for _ in range(max_tries):
        grid = rng.integers(0, 5, size=size)
        if is_feasible(grid):
            return grid
    raise AssertionError("could not sample a feasible grid")


@pytest.mark.parametrize("size", GRID_SIZES)
def test_random_valid_body_steps_100_times(size):
    rng = np.random.default_rng(SMOKE_SEED)
    grid = random_valid_grid(rng, size)
    assert is_feasible(grid)

    env = make_env(grid, seed=SMOKE_SEED)
    try:
        assert env.spec.max_episode_steps == MAX_EPISODE_STEPS
        n_act = env.action_space.shape[0]
        assert n_act > 0

        steps = 0
        for _ in range(100):
            action = rng.uniform(ACTION_LOW, ACTION_HIGH, size=n_act)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            assert np.all(np.isfinite(obs))
            assert np.isfinite(reward)
            if terminated or truncated:
                break
        # 100 steps is well inside the 500-step cap; a healthy body should not end early.
        assert steps == 100, f"episode ended after {steps} steps"

        com = robot_com(env)
        assert com.shape == (2,)
        assert np.all(np.isfinite(com))
    finally:
        env.close()


@pytest.mark.parametrize("size", GRID_SIZES)
def test_env_action_space_is_the_expected_range(size):
    """EvoGym actuators take a target volume ratio in [0.6, 1.6]; 1.0 is neutral."""
    rng = np.random.default_rng(SMOKE_SEED)
    grid = random_valid_grid(rng, size)
    env = make_env(grid, seed=SMOKE_SEED)
    try:
        assert env.spec.id == TASK
        assert np.allclose(env.action_space.low, ACTION_LOW)
        assert np.allclose(env.action_space.high, ACTION_HIGH)
    finally:
        env.close()
