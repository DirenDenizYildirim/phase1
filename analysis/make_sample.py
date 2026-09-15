"""Step 3: draw the morphology samples used for axis computation.

Writes (idempotently -- existing files are left alone):
  data/sample.parquet            600 decile-stratified + top-100 by true fitness, with grids
  data/sample_large_ids.parquet  3000 decile-stratified, IDs only, for a later run

Deciles are *rank-based* (equal-count groups), not value-based. The true-fitness distribution
piles up at the "did not move" floor -- over a quarter of all morphologies sit within 0.08 of
-5.0 -- so value-based bin edges would collide and several deciles would be empty or degenerate.
Equal-count groups give each stratum the same number of morphologies by construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from axes.common import BOUNDING_BOX, idx_to_grid  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = ROOT / "data" / "ground_truth.parquet"
SAMPLE = ROOT / "data" / "sample.parquet"
SAMPLE_LARGE = ROOT / "data" / "sample_large_ids.parquet"

# --- seeds (also recorded in notes/seeds.md) ---
SEED_SAMPLE = 12345        # the 600-morphology first-run sample
SEED_SAMPLE_LARGE = 54321  # the 3000-morphology later-run sample

N_DECILES = 10
PER_DECILE = 60            # 10 x 60 = 600
N_TOP = 100
N_LARGE_PER_DECILE = 300   # 10 x 300 = 3000


def grid_str(idx: int) -> str:
    """The 9-digit base-5 string for a morphology ID -- compact, readable, reversible."""
    return "".join(str(v) for v in idx_to_grid(idx, BOUNDING_BOX).flatten())


def assign_deciles(fitness: np.ndarray) -> np.ndarray:
    """Rank-based decile index 0..9, ties broken deterministically by position."""
    order = np.argsort(fitness, kind="stable")       # ascending; stable => deterministic
    ranks = np.empty(len(fitness), dtype=np.int64)
    ranks[order] = np.arange(len(fitness))
    return (ranks * N_DECILES) // len(fitness)


def stratified_sample(df: pd.DataFrame, per_decile: int, seed: int) -> np.ndarray:
    """Sample `per_decile` IDs from each decile, without replacement. Returns sorted IDs."""
    rng = np.random.default_rng(seed)
    picked = []
    for d in range(N_DECILES):
        pool = df.loc[df["decile"] == d, "id"].to_numpy()
        pool.sort()                                   # deterministic pool order
        if len(pool) < per_decile:
            raise ValueError(f"decile {d} has only {len(pool)} members")
        picked.append(rng.choice(pool, size=per_decile, replace=False))
    return np.sort(np.concatenate(picked))


def main() -> None:
    if SAMPLE.exists() and SAMPLE_LARGE.exists():
        print("both sample files already exist; nothing to do")
        return

    gt = pd.read_parquet(GROUND_TRUTH)
    gt["decile"] = assign_deciles(gt["true_fitness"].to_numpy())
    print(f"ground truth: {len(gt):,} morphologies")
    print(gt.groupby("decile")["true_fitness"].agg(["count", "min", "max"]).to_string())

    if not SAMPLE.exists():
        strat_ids = stratified_sample(gt, PER_DECILE, SEED_SAMPLE)
        top_ids = gt.nlargest(N_TOP, "true_fitness")["id"].to_numpy()

        keep = np.union1d(strat_ids, top_ids)
        out = gt[gt["id"].isin(keep)].copy()
        out["in_stratified"] = out["id"].isin(strat_ids)
        out["in_top100"] = out["id"].isin(top_ids)
        out["grid"] = [grid_str(int(i)) for i in out["id"]]
        out = out.sort_values("id").reset_index(drop=True)
        out = out[["id", "grid", "true_fitness", "decile", "in_stratified", "in_top100",
                   "raw_final", "raw_gen1"]]
        out.to_parquet(SAMPLE, index=False, compression="zstd")
        print(f"\nwrote {SAMPLE.name}: {len(out)} rows "
              f"({out['in_stratified'].sum()} stratified, {out['in_top100'].sum()} top-100, "
              f"{(out['in_stratified'] & out['in_top100']).sum()} in both)")
        print(f"  true_fitness range {out['true_fitness'].min():.4f} .. "
              f"{out['true_fitness'].max():.4f}")

    if not SAMPLE_LARGE.exists():
        large_ids = stratified_sample(gt, N_LARGE_PER_DECILE, SEED_SAMPLE_LARGE)
        large = gt[gt["id"].isin(large_ids)][["id", "true_fitness", "decile"]].copy()
        large = large.sort_values("id").reset_index(drop=True)
        large.to_parquet(SAMPLE_LARGE, index=False, compression="zstd")
        print(f"wrote {SAMPLE_LARGE.name}: {len(large)} rows (IDs only, for a later run)")


if __name__ == "__main__":
    main()
