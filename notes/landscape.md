# Whole-map checks: landscape smoothness and the noise ceiling

Both computed over all 1,305,840 morphologies with **no simulation at all**, via
`analysis/landscape_checks.py` (runs in well under a minute). Raw numbers in
`results/landscape_checks.json`.

The neighbour sweep needs no `is_connected` call. The published key set **is** the feasible set
(proved by exact set match in `tests/test_id_mapping.py`), so a neighbour is feasible iff its ID
is a key; and changing the voxel at row-major position `p` from value `v` to `t` shifts the base-5
ID by exactly `(t - v) * 5**(8 - p)`. The whole thing is 45 vectorised numpy passes.

---

## 1. Landscape smoothness

Predict each morphology's fitness by the **mean true fitness of its feasible one-voxel
neighbours**, then rank-correlate against truth.

| quantity | value |
|---|---|
| morphologies | 1,305,840 (all have ≥1 feasible neighbour) |
| feasible neighbours per morphology | mean 32.0, min 15, max 36 (of 36 candidates) |
| **Spearman, mean-neighbour vs truth** | **0.814** |
| Pearson, mean-neighbour vs truth | 0.830 |
| Spearman, *max*-neighbour vs truth | 0.614 |

**This is an oracle, not a ceiling for our axes.** It is handed every neighbour's ground-truth
fitness — information no training-free predictor has. Read it as: *how strongly does a body's
position in the design space constrain its fitness?* Answer: quite strongly, ρ = 0.814.

### The important caveat: almost all of that is coarse-scale

Recomputing the same statistic **within** each true-fitness decile:

| decile | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Spearman | +0.07 | +0.09 | +0.02 | +0.11 | +0.23 | +0.15 | +0.14 | +0.14 | +0.21 | **+0.48** |

Inside any decile but the top one, even a full neighbour oracle manages ρ ≤ 0.23. The 0.814
figure is driven overwhelmingly by separating broad bands of the landscape, not by ordering
morphologies within a band.

**What this means for the axes.** Stage 1's best model sits at CV Spearman 0.655 against an
oracle reference of 0.814, so there is real headroom — but that headroom is concentrated in
coarse separation, which the reach axes already do reasonably well. Fine-grained ranking looks
close to irreducible from body identity alone: the landscape is smooth at large scale and nearly
flat-to-noisy at small scale. A predictor chasing the last 0.15 is chasing the easier part of the
problem, not the hard one.

## 2. Noise ceiling — **unknown, and here is why**

`raw_results.npy` holds, for each morphology, a **single 1-D array of 300 floats**: the best-so-far
fitness after each generation of one controller-evolution run. Verified over a 200-morphology
sample — every value is a `(300,)` ndarray, never 2-D, never a list of runs.

**There is one attempt per morphology, so no split-half reliability can be computed. The noise
ceiling on ground truth is unknown.**

### What the map *does* tell us, and what it does not

The paper (§"Updating the Landscape") says fitness was re-estimated for 76,526 morphologies using
controllers discovered during the 10,000-generation co-optimization runs. Measured over the full
map:

| quantity | value |
|---|---|
| morphologies revised | **76,525** (5.86%) — paper says 76,526 |
| revised **upward** | 76,525 |
| revised **downward** | 0 |
| median revision | +0.743 |
| mean revision | +1.047 |
| max revision | +9.301 |
| Spearman(300-gen final, published true fitness) | 0.976 |

> **That 0.976 is NOT a reliability estimate and must not be reported as one.** True fitness is
> defined as the *better* of the 300-generation result and anything the co-optimization runs
> found, so the two quantities are dependent by construction — one is a componentwise maximum
> involving the other. A genuine split-half would need two independent attempts, which the
> published data does not contain.

What *is* sound to say: a single 300-generation run under-estimates achievable fitness for at
least 5.86% of morphologies, by a median of 0.74 and by as much as 9.30 — on a scale where the
whole population spans about 9.7. So single-run error is real and one-sided (never optimistic),
and published "true fitness" remains a best-of-attempts lower bound on what each body can do.

**Consequence for Stage 1's headline number.** Because the target is noisy by an unquantified
amount, our CV Spearman of 0.655 is an under-estimate of correlation with *achievable* fitness by
an unknown margin. We cannot say how much, and we do not guess.
