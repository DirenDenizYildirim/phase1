"""Step 4 driver: compute every axis for every sampled morphology.

Resumable and incremental by design -- the sandbox can vanish at any point:

* results already in `data/axes.parquet` are skipped on restart;
* the parquet is rewritten atomically every `CHECKPOINT_EVERY` completions, so at most that
  many morphologies of work is ever at risk;
* morphologies are farmed out to a process pool, one whole morphology per task, so the
  memoised sinusoid sweep is shared by axes 4c/4d/4e/4f within a worker.

Usage:
    python analysis/compute_axes.py [--sample data/sample.parquet] [--workers N] [--limit N]
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

from axes import efficiency, knockout, neighborhood, passive, reach, structural  # noqa: E402
from axes.common import grid_from_str  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = ROOT / "data" / "sample.parquet"
AXES_PARQUET = ROOT / "data" / "axes.parquet"
TIMING_MD = ROOT / "notes" / "timing.md"

# Order matters: `reach` populates the memoised sinusoid sweep that efficiency, knockout and
# neighborhood then reuse, so it must run before them.
AXIS_MODULES = (structural, passive, reach, efficiency, knockout, neighborhood)

CHECKPOINT_EVERY = 20


def compute_one(task):
    """Compute every axis for one morphology. Returns (row, per-axis seconds)."""
    mid, grid_str = task
    grid = grid_from_str(grid_str)
    row = {"id": int(mid)}
    times = {}
    for mod in AXIS_MODULES:
        t0 = time.perf_counter()
        row.update(mod.compute(grid))
        times[mod.AXIS_NAME] = time.perf_counter() - t0
    # Bound worker memory: the sweep cache is only useful within one morphology.
    from axes.sim import _sweep_cached
    _sweep_cached.cache_clear()
    return row, times


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def write_timing(times_by_axis: dict[str, list[float]], n_done: int, wall: float) -> None:
    lines = [
        "# Per-axis compute time",
        "",
        "Mean wall-clock seconds per morphology, measured inside the workers during the Step 4 run.",
        "",
        "| axis | mean s/morphology | median | max | share |",
        "|---|---|---|---|---|",
    ]
    total = sum(np.mean(v) for v in times_by_axis.values() if v)
    for name in (m.AXIS_NAME for m in AXIS_MODULES):
        v = times_by_axis.get(name) or []
        if not v:
            continue
        lines.append(f"| {name} | {np.mean(v):.3f} | {np.median(v):.3f} | {np.max(v):.3f} "
                     f"| {100 * np.mean(v) / total:.1f}% |")
    lines += [
        f"| **all axes** | **{total:.3f}** | | | 100% |",
        "",
        f"Morphologies completed: {n_done}. Total wall-clock for the run: {wall / 60:.1f} min "
        f"across {mp.cpu_count()} cores.",
        "",
        "The brief's target was under ~2 s per morphology per axis. Every axis meets that except "
        "the two that re-simulate derived bodies (`knockout`, `neighborhood`) and `reach`, whose "
        "16-episode sinusoid sweep is the headline probe. See notes/scope.md for the budget.",
    ]
    TIMING_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--out", type=Path, default=AXES_PARQUET)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--limit", type=int, default=None, help="only compute this many (for testing)")
    args = ap.parse_args()

    sample = pd.read_parquet(args.sample)
    done_rows = []
    if args.out.exists():
        done = pd.read_parquet(args.out)
        done_rows = done.to_dict("records")
        have = set(done["id"].tolist())
        print(f"resuming: {len(have)} morphologies already computed")
    else:
        have = set()

    todo = [(int(r.id), r.grid) for r in sample.itertuples() if int(r.id) not in have]
    if args.limit:
        todo = todo[:args.limit]
    print(f"to compute: {len(todo)} morphologies on {args.workers} workers")
    if not todo:
        print("nothing to do")
        return

    times_by_axis: dict[str, list[float]] = {}
    t_start = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for i, (row, times) in enumerate(pool.imap_unordered(compute_one, todo, chunksize=1), 1):
            done_rows.append(row)
            for k, v in times.items():
                times_by_axis.setdefault(k, []).append(v)
            if i % CHECKPOINT_EVERY == 0 or i == len(todo):
                atomic_write_parquet(pd.DataFrame(done_rows).sort_values("id"), args.out)
                elapsed = time.perf_counter() - t_start
                rate = elapsed / i
                print(f"  [{i}/{len(todo)}] checkpointed {len(done_rows)} rows | "
                      f"{rate:.1f}s/morph | eta {(len(todo) - i) * rate / 60:.1f} min",
                      flush=True)

    wall = time.perf_counter() - t_start
    atomic_write_parquet(pd.DataFrame(done_rows).sort_values("id"), args.out)
    write_timing(times_by_axis, len(done_rows), wall)
    print(f"done: {len(done_rows)} rows in {wall / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
