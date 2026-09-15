"""Two whole-map checks that need no simulation at all.

1. **Landscape smoothness.** For every morphology, predict its fitness from the mean true
   fitness of its feasible one-voxel neighbours, and rank-correlate that against truth. This is
   an *oracle* predictor -- it is handed every neighbour's ground-truth fitness, which no real
   predictor has -- so it is a reference point for how strongly a body's identity constrains its
   fitness, not a ceiling any cheap axis could be expected to hit.

2. **Noise ceiling.** Whether the published map contains repeated independent attempts per
   morphology, which would let us split them and correlate half against half.

The neighbour enumeration needs no `is_connected` call: the published key set *is* the feasible
set (verified in tests/test_id_mapping.py), so a neighbour is feasible iff its ID is a key.
Changing the voxel at row-major position p from value v to t shifts the base-5 ID by
(t - v) * 5**(8 - p), which makes the whole sweep a handful of vectorised passes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = ROOT / "data" / "ground_truth.parquet"
RAW_NPY = ROOT / "data" / "mc-landscape" / "results" / "raw_results.npy"
RESULTS = ROOT / "results"

N_CELLS = 9
N_TYPES = 5
N_IDS = N_TYPES ** N_CELLS          # 1,953,125


def neighbour_fitness_stats(ids: np.ndarray, fitness: np.ndarray):
    """Sum, max and count of feasible one-voxel neighbours' true fitness, per morphology."""
    lookup = np.full(N_IDS, np.nan, dtype=np.float64)
    lookup[ids] = fitness

    total = np.zeros(len(ids))
    count = np.zeros(len(ids), dtype=np.int32)
    best = np.full(len(ids), -np.inf)

    for p in range(N_CELLS):
        place = N_TYPES ** (N_CELLS - 1 - p)
        digit = (ids // place) % N_TYPES
        for t in range(N_TYPES):
            delta = (t - digit) * place
            changed = digit != t
            nbr = ids + delta
            vals = lookup[nbr]
            ok = changed & np.isfinite(vals)
            total[ok] += vals[ok]
            count[ok] += 1
            np.maximum(best, np.where(ok, vals, -np.inf), out=best)
    return total, count, best


def smoothness(gt: pd.DataFrame) -> dict:
    ids = gt["id"].to_numpy()
    y = gt["true_fitness"].to_numpy()
    total, count, best = neighbour_fitness_stats(ids, y)

    has = count > 0
    mean_nbr = np.full(len(ids), np.nan)
    mean_nbr[has] = total[has] / count[has]

    rho_mean = spearmanr(mean_nbr[has], y[has]).statistic
    rho_max = spearmanr(best[has], y[has]).statistic
    pearson_mean = float(np.corrcoef(mean_nbr[has], y[has])[0, 1])

    out = {
        "n_morphologies": int(len(ids)),
        "n_with_neighbours": int(has.sum()),
        "neighbours_per_morphology_mean": float(count.mean()),
        "neighbours_per_morphology_min": int(count.min()),
        "neighbours_per_morphology_max": int(count.max()),
        "spearman_mean_neighbour_vs_truth": float(rho_mean),
        "spearman_max_neighbour_vs_truth": float(rho_max),
        "pearson_mean_neighbour_vs_truth": pearson_mean,
    }

    # Same statistic restricted to the deciles, to show where locality does and does not hold.
    order = np.argsort(y, kind="stable")
    dec = np.empty(len(y), dtype=np.int64)
    dec[order] = np.arange(len(y)) * 10 // len(y)
    per_decile = {}
    for d in range(10):
        m = has & (dec == d)
        per_decile[d] = float(spearmanr(mean_nbr[m], y[m]).statistic)
    out["spearman_by_decile"] = per_decile
    return out


def noise_ceiling() -> dict:
    """Does the map hold repeated independent attempts per morphology?"""
    raw = np.load(RAW_NPY, allow_pickle=True).item()
    keys = list(raw.keys())
    sample = keys[:200]

    shapes = {np.shape(raw[k]) for k in sample}
    types = {type(raw[k]).__name__ for k in sample}
    n_values_per_key = {np.size(raw[k]) for k in sample}

    out = {
        "value_types": sorted(types),
        "value_shapes": sorted(str(s) for s in shapes),
        "values_per_morphology": sorted(int(v) for v in n_values_per_key),
        "n_morphologies": len(keys),
    }
    # A repeated-attempt layout would be 2-D (attempts x generations) or a list of runs.
    out["has_repeated_attempts"] = all(len(s) > 1 for s in shapes)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    gt = pd.read_parquet(GROUND_TRUTH)

    print("=== 1. Landscape smoothness (whole map, no simulation) ===")
    s = smoothness(gt)
    for k, v in s.items():
        if k != "spearman_by_decile":
            print(f"  {k}: {v}")
    print("  Spearman within each true-fitness decile:")
    for d, v in s["spearman_by_decile"].items():
        print(f"    decile {d}: {v:+.3f}")

    print("\n=== 2. Noise ceiling ===")
    n = noise_ceiling()
    for k, v in n.items():
        print(f"  {k}: {v}")

    # What the map DOES tell us about single-run reliability: the revision statistics.
    d = gt["true_fitness"].to_numpy() - gt["raw_final"].to_numpy()
    revised = np.abs(d) > 1e-9
    rev = {
        "n_revised_upward": int((d > 1e-9).sum()),
        "n_revised_downward": int((d < -1e-9).sum()),
        "fraction_revised": float(revised.mean()),
        "revision_median": float(np.median(d[revised])) if revised.any() else None,
        "revision_mean": float(d[revised].mean()) if revised.any() else None,
        "revision_max": float(d[revised].max()) if revised.any() else None,
        "spearman_raw_final_vs_true": float(spearmanr(gt["raw_final"], gt["true_fitness"]).statistic),
    }
    print("\n  Single-run vs revised estimate (the only reliability evidence available):")
    for k, v in rev.items():
        print(f"    {k}: {v}")

    payload = {"smoothness": s, "noise_ceiling": n, "revision_stats": rev}
    (RESULTS / "landscape_checks.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {RESULTS / 'landscape_checks.json'}")


if __name__ == "__main__":
    main()
