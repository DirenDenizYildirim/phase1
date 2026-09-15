"""Shared simulation machinery for the behavioural axes (4b-4f).

Nothing here trains anything. Every controller is a fixed open-loop signal: constant, uniform
random, or a sinusoid. No policy parameters are ever updated.

Conventions established empirically against EvoGym 2.0.0 / simulator v2.2.5 (see
notes/dataset.md):

* `env.action_space` is `Box(0.6, 1.6)` per actuator -- a target volume ratio, 1.0 neutral.
* Actions are ordered by the actuator voxels' **row-major flat index** into the grid, so
  `action[i]` drives the i-th actuator in a top-to-bottom, left-to-right scan.
* Grid **row 0 is the top** of the robot; row increases downward.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from axes.common import (ACTION_HIGH, ACTION_LOW, ACTION_NEUTRAL, H_ACT, make_env, robot_com,
                         robot_vel)

# --- probe configuration (fixed; also recorded in notes/seeds.md) ---
PROBE_SEED = 17            # matches the seed the ground-truth runs used
PROBE_STEPS = 300          # episode length for the reach probes
SETTLE_STEPS = 200         # passive-stability episode length
SETTLE_TAIL = 50           # window at the end of the settle episode for the velocity statistic
N_RANDOM_EPISODES = 8
HORIZON_SHORT = 100        # the ground truth's own episode length -- recorded alongside the full run

# A sinusoid spanning exactly the action space: 1.1 +/- 0.5 covers [0.6, 1.6].
SIN_CENTER = 0.5 * (ACTION_LOW + ACTION_HIGH)   # 1.1
SIN_AMPLITUDE = 0.5 * (ACTION_HIGH - ACTION_LOW)  # 0.5

# Cycles per step. Periods of 50, 20, 10 and 5 steps against a 100-step task horizon.
FREQUENCIES = (0.02, 0.05, 0.10, 0.20)
PHASE_PATTERNS = ("in_phase", "left_right", "top_bottom", "random")
RANDOM_PHASE_SEED = 90210   # fixed, so the "random" phase pattern is deterministic per body


def actuator_rowcol(grid: np.ndarray) -> np.ndarray:
    """(n_actuators, 2) array of (row, col) for each actuator, in action order."""
    grid = np.asarray(grid)
    flat = grid.flatten()
    idx = np.flatnonzero(flat >= H_ACT)
    return np.stack([idx // grid.shape[1], idx % grid.shape[1]], axis=1)


def phase_offsets(grid: np.ndarray, pattern: str) -> np.ndarray:
    """Per-actuator phase offsets in radians, in action order."""
    rc = actuator_rowcol(grid)
    n = len(rc)
    if pattern == "in_phase":
        return np.zeros(n)
    if pattern == "left_right":
        mid = (grid.shape[1] - 1) / 2.0
        return np.where(rc[:, 1] > mid, np.pi, 0.0)
    if pattern == "top_bottom":
        mid = (grid.shape[0] - 1) / 2.0
        return np.where(rc[:, 0] > mid, np.pi, 0.0)
    if pattern == "random":
        return np.random.default_rng(RANDOM_PHASE_SEED).uniform(0, 2 * np.pi, size=n)
    raise ValueError(f"unknown phase pattern {pattern!r}")


def sinusoid_actions(t: int, freq: float, phases: np.ndarray) -> np.ndarray:
    a = SIN_CENTER + SIN_AMPLITUDE * np.sin(2 * np.pi * freq * t + phases)
    return np.clip(a, ACTION_LOW, ACTION_HIGH)


def _episode(grid, action_of_t, steps, *, track_speed=False, track_action_cost=False):
    """Run one open-loop episode and return summary kinematics.

    Returns a dict with the net COM displacement over the full episode and over the ground
    truth's own 100-step horizon, plus optional speed and actuation-cost statistics.
    """
    env = make_env(grid, seed=PROBE_SEED)
    try:
        com0 = robot_com(env)
        dx_short = np.nan
        speeds = []
        act_abs = 0.0
        act_dev = 0.0
        steps_done = 0
        ended_early = False
        for t in range(steps):
            action = action_of_t(t)
            if track_action_cost:
                act_abs += float(np.abs(action).sum())
                act_dev += float(np.abs(action - ACTION_NEUTRAL).sum())
            _, _, terminated, truncated, _ = env.step(action)
            steps_done = t + 1
            if track_speed:
                v = robot_vel(env)
                speeds.append(float(np.mean(np.linalg.norm(v, axis=0))))
            if steps_done == HORIZON_SHORT:
                dx_short = float(robot_com(env)[0] - com0[0])
            if terminated or truncated:
                ended_early = True
                break
        com1 = robot_com(env)
        vel_end = robot_vel(env)
        out = {
            "dx": float(com1[0] - com0[0]),
            "dy": float(com1[1] - com0[1]),
            "dx_short": dx_short,
            "steps": float(steps_done),
            "ended_early": float(ended_early),
            "max_speed_end": float(np.max(np.linalg.norm(vel_end, axis=0))),
        }
        if track_speed:
            out["mean_speed"] = float(np.mean(speeds)) if speeds else 0.0
        if track_action_cost:
            out["action_abs_sum"] = act_abs
            out["action_dev_sum"] = act_dev
        return out
    finally:
        env.close()


def run_settle(grid) -> dict[str, float]:
    """Passive stability: hold every actuator at neutral and let the body settle."""
    env = make_env(grid, seed=PROBE_SEED)
    try:
        n = env.action_space.shape[0]
        neutral = np.full(n, ACTION_NEUTRAL)
        com0 = robot_com(env)
        tail_max_speed = 0.0
        tail_speeds = []
        for t in range(SETTLE_STEPS):
            _, _, terminated, truncated, _ = env.step(neutral)
            if t >= SETTLE_STEPS - SETTLE_TAIL:
                speeds = np.linalg.norm(robot_vel(env), axis=0)
                tail_max_speed = max(tail_max_speed, float(np.max(speeds)))
                tail_speeds.append(float(np.mean(speeds)))
            if terminated or truncated:
                break
        com1 = robot_com(env)
        return {
            "settle_com_y_drift": float(com1[1] - com0[1]),
            "settle_com_x_drift": float(com1[0] - com0[0]),
            # EvoGym's soft bodies never come fully to rest -- speeds decay over the first ~50
            # steps then plateau at a body-dependent jitter level. The max over the tail window
            # is a noisy order statistic, so the mean is reported alongside it.
            "settle_max_speed_tail": tail_max_speed,
            "settle_mean_speed_tail": float(np.mean(tail_speeds)) if tail_speeds else 0.0,
        }
    finally:
        env.close()


def run_random_episodes(grid, n_episodes: int = N_RANDOM_EPISODES) -> list[dict]:
    """Uniform random actions over the actuator range, one RNG stream seeded per episode."""
    out = []
    for e in range(n_episodes):
        rng = np.random.default_rng(PROBE_SEED * 1000 + e)
        n = len(actuator_rowcol(grid))

        def act(t, rng=rng, n=n):
            return rng.uniform(ACTION_LOW, ACTION_HIGH, size=n)

        out.append(_episode(grid, act, PROBE_STEPS, track_speed=True))
    return out


def run_sinusoid(grid, freq: float, pattern: str, steps: int = PROBE_STEPS) -> dict:
    phases = phase_offsets(np.asarray(grid), pattern)
    return _episode(grid, lambda t: sinusoid_actions(t, freq, phases), steps,
                    track_action_cost=True)


def _sweep(grid: np.ndarray, freqs=FREQUENCIES, patterns=PHASE_PATTERNS) -> list[dict]:
    res = []
    for freq in freqs:
        for pattern in patterns:
            r = run_sinusoid(grid, freq, pattern)
            r["freq"] = freq
            r["pattern"] = pattern
            res.append(r)
    return res


@lru_cache(maxsize=256)
def _sweep_cached(grid_bytes: bytes, shape: tuple[int, int]):
    grid = np.frombuffer(grid_bytes, dtype=np.int64).reshape(shape)
    return tuple(_sweep(grid))


def sinusoid_sweep(grid: np.ndarray) -> list[dict]:
    """Full 4x4 sinusoid sweep, memoised so axes 4c/4d/4e can share one set of episodes."""
    grid = np.ascontiguousarray(np.asarray(grid, dtype=np.int64))
    return [dict(r) for r in _sweep_cached(grid.tobytes(), grid.shape)]


# --- reduced sweep, used by the expensive axes (4e knockout, 4f neighbourhood) ---
# Running the full 16-episode sweep for every knockout and every neighbour would cost ~30x more
# simulation than the whole rest of the pipeline. This reduced set follows a rule fixed in advance
# -- hold the frequency at the most productive value and sweep the symmetry-breaking phase
# patterns -- rather than the argmax over subsets, which would overfit the pilot. Measured on a
# 24-morphology pilot spanning the fitness range, its best-dx ranking correlates with the full
# 16-episode sweep at Spearman 0.950. Phase pattern carries nearly all the signal: varying
# frequency at a fixed pattern reaches only ~0.6. See notes/scope.md.
REDUCED_SWEEP: tuple[tuple[float, str], ...] = (
    (0.10, "left_right"),
    (0.10, "top_bottom"),
    (0.10, "random"),
)

def reduced_sweep_best_dx(grid: np.ndarray) -> float:
    """Best net displacement over the reduced sweep -- the cheap stand-in for a full sweep."""
    return max(run_sinusoid(grid, f, p)["dx"] for f, p in REDUCED_SWEEP)


def best_of(results: list[dict], key: str = "dx") -> dict:
    """The sweep entry with the largest net forward displacement.

    Ties break on (freq, pattern) order, which is fixed, so this is deterministic.
    """
    return max(results, key=lambda r: r[key])
