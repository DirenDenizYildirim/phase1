"""How does rand_dx_max's correlation buy itself with compute?

Runs 16 random-actuation episodes per morphology and records each episode's net displacement,
so the axis can be recomputed over nested prefixes (2, 4, 8, 16 episodes) without re-simulating.

Two protocols are run side by side:

* `current`  -- what Stage 1 used: 300 simulator steps, a fresh action every step.
* `matched`  -- what the ground truth actually does (confirmed against the paper and by
  replicating the upstream wrapper stack): 500 simulator steps with the action held for 5 steps,
  i.e. 100 controller queries. Stage 1's probes matched neither the horizon nor the control rate.

Episode seeds are `PROBE_SEED * 1000 + episode_index`, the same scheme axes/sim.py uses, so the
first 8 `current` episodes reproduce the values already in data/axes.parquet.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from axes.common import (ACTION_HIGH, ACTION_LOW, grid_from_str, make_env,  # noqa: E402
                         robot_com)
from axes.sim import PROBE_SEED  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "episode_ablation.parquet"

N_EPISODES = 16
PROTOCOLS = {
    # name:      (n_simulator_steps, action_hold)
    "current":   (300, 1),
    "matched":   (500, 5),
}
CHECKPOINT_EVERY = 20


def one_episode(grid, seed: int, n_steps: int, hold: int) -> float:
    """Net COM x-displacement under uniform random actions, resampled every `hold` steps."""
    env = make_env(grid, seed=PROBE_SEED)
    try:
        rng = np.random.default_rng(seed)
        n = env.action_space.shape[0]
        x0 = robot_com(env)[0]
        # No pre-loop draw: t=0 always satisfies `t % hold == 0`, so seeding one here would
        # shift the whole random stream by one action and break agreement with axes/sim.py.
        action = None
        for t in range(n_steps):
            if t % hold == 0:
                action = rng.uniform(ACTION_LOW, ACTION_HIGH, size=n)
            _, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        return float(robot_com(env)[0] - x0)
    finally:
        env.close()


def compute_one(task):
    mid, gs = task
    grid = grid_from_str(gs)
    row = {"id": int(mid)}
    for pname, (n_steps, hold) in PROTOCOLS.items():
        for e in range(N_EPISODES):
            row[f"{pname}_dx_{e:02d}"] = one_episode(grid, PROBE_SEED * 1000 + e, n_steps, hold)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, default=ROOT / "data" / "sample.parquet")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sample = pd.read_parquet(args.sample)
    sample = sample[sample["in_stratified"]]        # the ablation follows the analysis sample
    rows, have = [], set()
    if OUT.exists():
        done = pd.read_parquet(OUT)
        rows = done.to_dict("records")
        have = set(done["id"].tolist())
        print(f"resuming: {len(have)} already done")

    todo = [(int(r.id), r.grid) for r in sample.itertuples() if int(r.id) not in have]
    if args.limit:
        todo = todo[:args.limit]
    print(f"to compute: {len(todo)} morphologies x {N_EPISODES} episodes x "
          f"{len(PROTOCOLS)} protocols on {args.workers} workers")
    if not todo:
        print("nothing to do")
        return

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(compute_one, todo, chunksize=1), 1):
            rows.append(row)
            if i % CHECKPOINT_EVERY == 0 or i == len(todo):
                tmp = OUT.with_suffix(".parquet.tmp")
                pd.DataFrame(rows).sort_values("id").to_parquet(tmp, index=False,
                                                                compression="zstd")
                os.replace(tmp, OUT)
                el = time.perf_counter() - t0
                print(f"  [{i}/{len(todo)}] {el / i:.1f}s/morph  "
                      f"eta {(len(todo) - i) * el / i / 60:.1f} min", flush=True)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
