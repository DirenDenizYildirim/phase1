"""Step 5: does any cheap axis carry signal about achievable fitness?

Rerunnable on a larger sample: it reads whatever is in `data/axes.parquet` and joins it to
`data/sample.parquet`, so pointing it at the 3000-morphology run needs no code change.

Methodology notes that matter for reading the output:

* The sample is 600 decile-stratified morphologies **plus** the true top 100. The top-100 block
  is a deliberate over-sample of the extreme tail, so including it would inflate every
  correlation. **All correlation and cross-validation results are computed on the 600
  stratified morphologies only.** The top 100 are used solely for the top-100 recall analysis,
  which needs them present by construction.
* Decile-stratified sampling is rank-uniform, so Spearman on it estimates the population
  Spearman without the floor-effect ties that dominate a uniform random sample.
* `HistGradientBoostingRegressor` consumes NaN natively; Ridge gets median imputation. Knockout
  fractions are NaN when the intact body barely moves (see axes/knockout.py) -- those are
  genuinely missing, not zero, and are never filled in with a fabricated value.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SEED = 20240517
N_BOOTSTRAP = 2000
N_FOLDS = 5

# Bookkeeping columns: diagnostics of the probe, not descriptors of the morphology.
DROP_COLS = {"id", "ko_n_tested", "nbr_n_simulated", "eff_action_abs_sum", "eff_action_dev_sum"}

# Axes computable with no simulation at all -- this is the honest structural-only set.
# The neighbourhood *feasibility* counts are pure combinatorics over one-voxel edits, so they
# belong here even though the rest of the neighbourhood axis needs simulation.
STRUCTURAL_COLS = [
    "n_voxels", "n_actuators", "actuator_fraction", "h_actuator_fraction", "aspect_ratio",
    "compactness", "h_symmetry", "modularity", "n_components",
    "nbr_feasible_fraction", "nbr_n_feasible",
]

AXIS_FAMILY = {
    "structural": STRUCTURAL_COLS,
    "passive": ["settle_com_y_drift", "settle_com_x_drift", "settle_max_speed_tail",
                "settle_mean_speed_tail"],
    "reach": ["rand_dx_mean", "rand_dx_max", "rand_dx_std", "rand_mean_speed", "sin_best_dx",
              "sin_best_dx_short", "sin_best_freq", "sin_best_pattern_id", "sin_dx_mean",
              "sin_dx_std", "sin_dx_range"],
    "efficiency": ["eff_raw", "eff_raw_log", "eff_dev", "eff_dev_log"],
    "knockout": ["ko_frac_mean", "ko_frac_min", "ko_dx_delta_mean", "ko_dx_delta_worst"],
    "neighborhood": ["nbr_dx_mean", "nbr_dx_max", "nbr_dx_std", "nbr_dx_mean_minus_self",
                     "nbr_dx_max_minus_self"],
}


# --------------------------------------------------------------------------- data

def load(sample_path: Path, axes_path: Path) -> tuple[pd.DataFrame, list[str]]:
    sample = pd.read_parquet(sample_path)
    axes = pd.read_parquet(axes_path)
    df = sample.merge(axes, on="id", how="inner")
    cols = [c for c in axes.columns if c not in DROP_COLS]

    dropped = [c for c in cols if df[c].nunique(dropna=True) <= 1]
    if dropped:
        print(f"dropping zero-variance axes: {dropped}")
        cols = [c for c in cols if c not in dropped]
    return df, cols


# --------------------------------------------------------------- per-axis correlation

def bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray, rng, n=N_BOOTSTRAP):
    """Percentile 95% CI for Spearman rho, resampling pairs with replacement."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        return np.nan, np.nan, np.nan, len(x)
    rho = spearmanr(x, y).statistic
    idx = rng.integers(0, len(x), size=(n, len(x)))
    boots = np.array([spearmanr(x[i], y[i]).statistic for i in idx])
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rho, lo, hi, len(x)


def per_axis_correlations(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    y = df["true_fitness"].to_numpy()
    rows = []
    for c in cols:
        rho, lo, hi, n = bootstrap_spearman_ci(df[c].to_numpy(dtype=float), y, rng)
        fam = next((k for k, v in AXIS_FAMILY.items() if c in v), "other")
        rows.append({"axis": c, "family": fam, "spearman": rho, "ci_lo": lo, "ci_hi": hi,
                     "n": n, "abs_spearman": abs(rho) if np.isfinite(rho) else np.nan,
                     "ci_excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))})
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------------- modelling

def make_models():
    return {
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               Ridge(alpha=1.0, random_state=None)),
        "gbt": HistGradientBoostingRegressor(random_state=SEED, max_iter=300,
                                             learning_rate=0.06, max_depth=4),
    }


def cv_predict(model, X: np.ndarray, y: np.ndarray):
    """Pooled out-of-fold predictions from a fresh clone per fold."""
    from sklearn.base import clone
    oof = np.full(len(y), np.nan)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in kf.split(X):
        m = clone(model).fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def score(y, pred) -> dict:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"cv_spearman": float(spearmanr(pred, y).statistic),
            "cv_r2": float(1 - ss_res / ss_tot)}


def gbt_permutation_importance(X, y, feature_names):
    """Per-fold permutation importance measured on the held-out fold, then averaged."""
    from sklearn.base import clone
    model = make_models()["gbt"]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    acc = np.zeros(X.shape[1])
    for tr, te in kf.split(X):
        m = clone(model).fit(X[tr], y[tr])
        r = permutation_importance(m, X[te], y[te], n_repeats=10,
                                   random_state=SEED, scoring="r2")
        acc += r.importances_mean
    return (pd.DataFrame({"axis": feature_names, "perm_importance": acc / N_FOLDS})
            .sort_values("perm_importance", ascending=False).reset_index(drop=True))


# ----------------------------------------------------------------------- top-100 recall

def top100_recall(full: pd.DataFrame, cols: list[str], gbt_oof: np.ndarray | None) -> pd.DataFrame:
    """Fraction of the true top-100 landing in each axis's top quartile of the pooled sample.

    Computed on the full 700-row sample (the 100 must be present to be recalled). The chance
    rate is 0.25 by construction, so anything near 0.25 is no signal at all.
    """
    is_top = full["in_top100"].to_numpy()
    n_q = int(np.ceil(0.25 * len(full)))
    rows = []

    def recall_of(values, name, family):
        v = np.asarray(values, dtype=float)
        ok = np.isfinite(v)
        # Missing values cannot be in the top quartile; rank them last.
        vv = np.where(ok, v, -np.inf)
        top_idx = np.argsort(-vv, kind="stable")[:n_q]
        in_top_q = np.zeros(len(full), dtype=bool)
        in_top_q[top_idx] = True
        rows.append({"axis": name, "family": family,
                     "top100_recall": float(in_top_q[is_top].mean())})

    for c in cols:
        fam = next((k for k, v in AXIS_FAMILY.items() if c in v), "other")
        # An axis can point either way; credit the orientation that correlates positively.
        r = spearmanr(full[c].to_numpy(dtype=float), full["true_fitness"].to_numpy(),
                      nan_policy="omit").statistic
        recall_of(full[c].to_numpy(dtype=float) * (1 if (r or 0) >= 0 else -1), c, fam)
    if gbt_oof is not None:
        recall_of(gbt_oof, "GBT (all axes, out-of-fold)", "model")
    return pd.DataFrame(rows).sort_values("top100_recall", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- plots

def plot_correlation_bars(corr: pd.DataFrame, path: Path):
    d = corr.dropna(subset=["spearman"]).sort_values("spearman")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.26 * len(d))))
    colors = {"structural": "#4C72B0", "passive": "#DD8452", "reach": "#55A868",
              "efficiency": "#C44E52", "knockout": "#8172B3", "neighborhood": "#937860",
              "other": "#999999"}
    y = np.arange(len(d))
    ax.barh(y, d["spearman"], color=[colors.get(f, "#999") for f in d["family"]])
    ax.errorbar(d["spearman"], y,
                xerr=[d["spearman"] - d["ci_lo"], d["ci_hi"] - d["spearman"]],
                fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
    ax.set_yticks(y); ax.set_yticklabels(d["axis"], fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman rho with true fitness (bootstrap 95% CI)")
    ax.set_title("Per-axis rank correlation with achievable fitness\n"
                 "(600 decile-stratified morphologies)", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_heatmap(df: pd.DataFrame, cols: list[str], path: Path):
    m = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.30 * len(cols) + 3, 0.30 * len(cols) + 2.5))
    im = ax.imshow(m, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=6)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=6)
    ax.set_title("Pairwise Spearman between axes (redundancy)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_pred_scatter(y, pred, stats, path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, pred, s=10, alpha=0.5, edgecolor="none")
    lo = min(y.min(), pred.min()); hi = max(y.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("true fitness"); ax.set_ylabel("out-of-fold GBT prediction")
    ax.set_title(f"GBT, all axes, 5-fold CV\nSpearman {stats['cv_spearman']:.3f}, "
                 f"R2 {stats['cv_r2']:.3f}", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_small_multiples(df: pd.DataFrame, cols: list[str], path: Path):
    n = len(cols); ncol = 6; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 1.9 * nrow))
    y = df["true_fitness"].to_numpy()
    for ax, c in zip(axes.flat, cols):
        ax.scatter(df[c], y, s=4, alpha=0.35, edgecolor="none")
        r = spearmanr(df[c].to_numpy(dtype=float), y, nan_policy="omit").statistic
        ax.set_title(f"{c}\nrho={r:.2f}", fontsize=6)
        ax.tick_params(labelsize=5)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Each axis vs true fitness (600 decile-stratified morphologies)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, default=ROOT / "data" / "sample.parquet")
    ap.add_argument("--axes", type=Path, default=ROOT / "data" / "axes.parquet")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    full, cols = load(args.sample, args.axes)
    strat = full[full["in_stratified"]].reset_index(drop=True)
    print(f"joined rows: {len(full)}  (stratified {len(strat)}, top100 {int(full['in_top100'].sum())})")
    print(f"axes analysed: {len(cols)}")

    out: dict = {"n_joined": len(full), "n_stratified": len(strat),
                 "n_top100": int(full["in_top100"].sum()), "n_axes": len(cols), "axes": cols}

    # --- 1. per-axis correlation, on the stratified sample only ---
    corr = per_axis_correlations(strat, cols)
    corr.to_csv(RESULTS / "axis_correlations.csv", index=False)
    print("\n=== per-axis Spearman with true fitness (600 stratified) ===")
    print(corr[["axis", "family", "spearman", "ci_lo", "ci_hi", "n",
                "ci_excludes_zero"]].to_string(index=False))

    # --- 2. redundancy ---
    red = strat[cols].corr(method="spearman")
    red.to_csv(RESULTS / "axis_redundancy.csv")
    ab = red.abs().where(~np.eye(len(cols), dtype=bool))
    pairs = (ab.stack().sort_values(ascending=False)
             .drop_duplicates().head(15).reset_index())
    pairs.columns = ["axis_a", "axis_b", "abs_spearman"]
    print("\n=== most redundant axis pairs ===")
    print(pairs.to_string(index=False))
    out["top_redundant_pairs"] = pairs.to_dict("records")

    # --- 3. models, 5-fold CV on the stratified sample ---
    y = strat["true_fitness"].to_numpy()
    groups = {"all": cols,
              "structural_only": [c for c in cols if c in STRUCTURAL_COLS],
              "simulation_only": [c for c in cols if c not in STRUCTURAL_COLS]}
    model_results = {}
    gbt_oof_strat = None
    for gname, gcols in groups.items():
        X = strat[gcols].to_numpy(dtype=float)
        for mname, model in make_models().items():
            oof = cv_predict(model, X, y)
            s = score(y, oof)
            model_results[f"{mname}|{gname}"] = {**s, "n_features": len(gcols)}
            if mname == "gbt" and gname == "all":
                gbt_oof_strat = oof
    print("\n=== 5-fold CV (600 stratified) ===")
    print(f"{'model|features':<28} {'n_feat':>6} {'CV Spearman':>12} {'CV R2':>8}")
    for k, v in model_results.items():
        print(f"{k:<28} {v['n_features']:>6} {v['cv_spearman']:>12.3f} {v['cv_r2']:>8.3f}")
    out["models"] = model_results

    # --- 4. permutation importance for the GBT ---
    perm = gbt_permutation_importance(strat[cols].to_numpy(dtype=float), y, cols)
    perm.to_csv(RESULTS / "gbt_permutation_importance.csv", index=False)
    print("\n=== GBT permutation importance (top 12, held-out folds) ===")
    print(perm.head(12).to_string(index=False))

    # --- 5. top-100 recall, on the full 700-row pool ---
    Xf = full[cols].to_numpy(dtype=float)
    gbt_oof_full = cv_predict(make_models()["gbt"], Xf, full["true_fitness"].to_numpy())
    rec = top100_recall(full, cols, gbt_oof_full)
    rec.to_csv(RESULTS / "top100_recall.csv", index=False)
    print("\n=== top-100 recall in the top quartile (chance = 0.25) ===")
    print(rec.head(15).to_string(index=False))
    out["top100_recall_chance"] = 0.25

    # --- 6. plots ---
    plot_correlation_bars(corr, RESULTS / "axis_correlations.png")
    plot_heatmap(strat, cols, RESULTS / "axis_redundancy_heatmap.png")
    plot_pred_scatter(y, gbt_oof_strat, model_results["gbt|all"],
                      RESULTS / "gbt_pred_vs_true.png")
    plot_small_multiples(strat, cols, RESULTS / "axes_vs_truth.png")

    (RESULTS / "analysis_summary.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote CSVs, plots and analysis_summary.json to {RESULTS}")


if __name__ == "__main__":
    main()
