"""How does rand_dx_max's correlation with true fitness buy itself with compute?

Reads `data/episode_ablation.parquet` (16 per-episode displacements per morphology, under two
protocols) and reports, for nested prefixes of 2/4/8/16 episodes:

* Spearman of `rand_dx_max` (and `rand_dx_mean`) against true fitness, with bootstrap 95% CIs;
* the simulator-step cost that buys it.

The two protocols are `current` (300 steps, fresh action every step -- what Stage 1 used) and
`matched` (500 steps, action held 5 steps -- what the ground truth actually does).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SEED = 20240517
N_BOOTSTRAP = 2000
PREFIXES = (1, 2, 4, 8, 16)
PROTOCOL_STEPS = {"current": 300, "matched": 500}


def boot_ci(x, y, rng, n=N_BOOTSTRAP):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho = spearmanr(x, y).statistic
    idx = rng.integers(0, len(x), size=(n, len(x)))
    b = np.array([spearmanr(x[i], y[i]).statistic for i in idx])
    b = b[np.isfinite(b)]
    return float(rho), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> None:
    ab = pd.read_parquet(ROOT / "data" / "episode_ablation.parquet")
    sample = pd.read_parquet(ROOT / "data" / "sample.parquet")
    df = ab.merge(sample[["id", "true_fitness"]], on="id")
    y = df["true_fitness"].to_numpy()
    print(f"morphologies: {len(df)}")

    rng = np.random.default_rng(SEED)
    rows = []
    for proto, steps in PROTOCOL_STEPS.items():
        for k in PREFIXES:
            cols = [f"{proto}_dx_{e:02d}" for e in range(k)]
            if not all(c in df.columns for c in cols):
                continue
            block = df[cols].to_numpy()
            for stat, vals in (("max", block.max(axis=1)), ("mean", block.mean(axis=1))):
                rho, lo, hi = boot_ci(vals, y, rng)
                rows.append({"protocol": proto, "statistic": stat, "episodes": k,
                             "sim_steps": k * steps, "spearman": rho,
                             "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "episode_ablation.csv", index=False)

    for proto in PROTOCOL_STEPS:
        sub = out[(out.protocol == proto)]
        if sub.empty:
            continue
        print(f"\n=== {proto} ({PROTOCOL_STEPS[proto]} steps/episode) ===")
        print(f"{'stat':<5} {'eps':>4} {'sim steps':>10} {'spearman':>9} {'95% CI':>18}")
        for r in sub.itertuples():
            print(f"{r.statistic:<5} {r.episodes:>4} {r.sim_steps:>10} {r.spearman:>9.3f}"
                  f"   [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")

    # --- paired comparisons ---
    # Overlapping marginal CIs do not settle whether two protocols differ, because both are
    # measured on the SAME morphologies. Bootstrap the difference of the two Spearmans instead.
    def col_max(proto, k):
        return df[[f"{proto}_dx_{e:02d}" for e in range(k)]].to_numpy().max(axis=1)

    def paired_diff(a, b, n=4000):
        d0 = spearmanr(a, y).statistic - spearmanr(b, y).statistic
        idx = rng.integers(0, len(y), size=(n, len(y)))
        d = np.array([spearmanr(a[i], y[i]).statistic - spearmanr(b[i], y[i]).statistic
                      for i in idx])
        lo, hi = np.percentile(d, [2.5, 97.5])
        return float(d0), float(lo), float(hi)

    paired = []
    for k in PREFIXES:
        d0, lo, hi = paired_diff(col_max("matched", k), col_max("current", k))
        paired.append({"comparison": f"matched({k} ep) - current({k} ep)", "kind": "equal episodes",
                       "diff": d0, "ci_lo": lo, "ci_hi": hi})
    # equal simulator-step budget: matched costs 500/episode, current 300, so k vs 2k is close
    for pk, ck in ((1, 2), (4, 8), (8, 16)):
        d0, lo, hi = paired_diff(col_max("matched", pk), col_max("current", ck))
        paired.append({"comparison": f"matched({pk} ep, {500*pk} steps) - "
                                     f"current({ck} ep, {300*ck} steps)",
                       "kind": "~equal cost", "diff": d0, "ci_lo": lo, "ci_hi": hi})
    for a, b in ((2, 1), (4, 2), (8, 4), (16, 8)):
        d0, lo, hi = paired_diff(col_max("current", a), col_max("current", b))
        paired.append({"comparison": f"current: {b} -> {a} episodes", "kind": "doubling",
                       "diff": d0, "ci_lo": lo, "ci_hi": hi})

    pdf = pd.DataFrame(paired)
    pdf["significant"] = (pdf.ci_lo > 0) | (pdf.ci_hi < 0)
    pdf.to_csv(RESULTS / "episode_ablation_paired.csv", index=False)
    print("\n=== paired bootstrap on the Spearman difference (same 600 morphologies) ===")
    print(f"{'comparison':<52} {'diff':>7} {'95% CI':>20}  sig")
    for r in pdf.itertuples():
        print(f"{r.comparison:<52} {r.diff:>+7.3f}   [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  "
              f"{'yes' if r.significant else 'no'}")

    # --- plot: correlation vs simulator-step cost ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {("current", "max"): ("#4C72B0", "-", "o"),
              ("current", "mean"): ("#4C72B0", "--", "s"),
              ("matched", "max"): ("#C44E52", "-", "o"),
              ("matched", "mean"): ("#C44E52", "--", "s")}
    for (proto, stat), (c, ls, mk) in styles.items():
        sub = out[(out.protocol == proto) & (out.statistic == stat)].sort_values("sim_steps")
        if sub.empty:
            continue
        ax.errorbar(sub["sim_steps"], sub["spearman"],
                    yerr=[sub["spearman"] - sub["ci_lo"], sub["ci_hi"] - sub["spearman"]],
                    color=c, linestyle=ls, marker=mk, capsize=3, ms=5, lw=1.4,
                    label=f"{proto} / rand_dx_{stat}")
    ax.set_xscale("log")
    ax.set_xlabel("simulator steps spent per morphology (log scale)")
    ax.set_ylabel("Spearman with true fitness")
    ax.set_title("What more random-actuation episodes buy\n(600 decile-stratified morphologies, "
                 "bootstrap 95% CI)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(RESULTS / "episode_ablation.png", dpi=150); plt.close(fig)

    (RESULTS / "episode_ablation.json").write_text(
        json.dumps({"curve": rows, "paired": paired}, indent=2))
    print(f"\nwrote episode_ablation.{{csv,png,json}} to {RESULTS}")


if __name__ == "__main__":
    main()
