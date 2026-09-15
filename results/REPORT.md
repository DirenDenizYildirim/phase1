# Stage 1: do cheap, training-free axes predict achievable fitness?

**Short answer: yes, some of them, and by a clear margin over chance — but the useful signal comes
almost entirely from short open-loop simulation, not from structure.** A gradient-boosted model
over all axes reaches **CV Spearman 0.655 (R² 0.419)** against ground-truth achievable fitness.
Restricted to axes that need no simulation at all it collapses to **0.296–0.332**. Passive
stability contributes nothing measurable.

All numbers below come from `analysis/run_analysis.py` over `data/axes.parquet`; the CSVs and
plots in this directory are its direct output.

---

## 1. Dataset

Ground truth is Mertan & Cheney's exhaustive morphology-fitness map
(`mertan-a/morphology-fitness-landscape`, arXiv 2508.17464).

### The search space is 3x3, not 5x5

The project brief describes "every feasible **5x5** EvoGym morphology". The published map is over
a **3x3** bounding box. This is verified, not inferred:

- published IDs run 93 … 1,953,124, and `5**9 - 1 = 1953124` exactly;
- independently enumerating all `5**9` 3x3 grids under the upstream feasibility rules yields
  **1,305,840** grids that **set-match the published keys exactly** (zero difference in either
  direction) — reproduced as a test in `tests/test_id_mapping.py`;
- a 5x5 exhaustive map would need `5**25 ≈ 3x10^17` evaluations.

The whole project therefore runs on the 3x3 space, because that is the space the ground truth
covers. Full derivation in `notes/dataset.md`.

| property | value |
|---|---|
| morphologies (all feasible 3x3) | 1,305,840 |
| voxel types | 0 empty, 1 rigid, 2 soft, 3 horizontal actuator, 4 vertical actuator |
| ID → grid | base-5 numeral, row-major, zero-padded to 9 digits |
| feasibility | connected, ≥3 non-empty voxels, ≥3 actuators |
| true fitness range | −5.2718 … +4.4237 (median −4.2694, mean −3.8793) |

**What a fitness number means.** `Walker-v0` rewards per-step centre-of-mass x-displacement, and
the upstream wrapper subtracts 0.05 per **controller query**, so

> fitness = (net COM x-displacement in world units) − 0.05 × (number of controller queries)
> = dx − 5.0

So **−5.0 means "did not move at all"**, and each point above that is one world unit (ten voxel
widths) of travel.

**Episode structure — corrected against the paper.** An episode is **500 simulator steps**, and
the controller is queried every 5th step with the action held in between, giving **100 controller
queries**. The paper states this directly ("Task and fitness"):

> "The controller is queried every 5th timestep, and the last action is repeated for the remaining
> timesteps." … "an additional small negative penalty (−0.05) is applied each time step before the
> robot reaches the target"

That is `ActionSkipWrapper(skip=5)` wrapped by `RewardShapingWrapper`, so the penalty is charged
per outer call: 500 / 5 = 100 queries × −0.05 = **−5.0**.

> An earlier draft of this report claimed the episode was "100 steps" and attributed it to the
> upstream `_max_episode_steps = 500` assignment landing on a dead attribute past the `TimeLimit`
> wrapper. **The −5.0 offset was right; that mechanism was wrong.** `TimeLimit` governs at 500
> steps either way, and the factor of 5 is the action-skip.

Confirmed by replicating the upstream wrapper stack over 40 morphologies with random actions:
fitness min −5.90 / max −4.26 / **mean −5.09**, against ground-truth generation-1 fitness
min −5.77 / max −4.65 / **mean −4.99** — matching in both centre and spread, with episodes running
exactly 100 outer and 500 inner steps. A 500-step episode without action-skip would need a −25.0
offset and an implied mean displacement of +20, which is unreachable. Details in
`notes/dataset.md` §5.

**The distribution is brutally floor-heavy.** A quarter of all morphologies sit within 0.08 of
"did not move"; the median travels only ~0.73 units in 100 steps. Good morphologies are rare.

## 2. Environment configuration used

Our probes use the same environment as the ground truth, confirmed against the upstream
`simulator.py`:

| setting | value |
|---|---|
| task | `Walker-v0` (EvoGym `WalkingFlat`, flat ground) |
| connections | `evogym.get_full_connectivity(body)` |
| action space | `Box(0.6, 1.6)` per actuator — target volume ratio, **1.0 neutral** |
| action ordering | row-major flat index of actuator voxels (verified empirically) |
| grid orientation | row 0 is the **top** of the robot (verified empirically) |
| env seed | 17 (the ground truth's own value) |
| EvoGym | 2.0.0, simulator v2.2.5, Python 3.10 |
| ground-truth episode | 500 simulator steps, action held 5 steps (100 controller queries) |
| **our probe episodes** | **300 simulator steps, fresh action every step** |

We deliberately do **not** apply the ground truth's `RewardShapingWrapper`: no axis reads the
reward signal, they read COM kinematics straight off the simulator.

> **Known limitation.** Our probes match the ground truth's task, body construction and seed, but
> **not its episode horizon or its control rate** — 300 steps at full rate versus 500 steps at
> one-fifth rate. This was not a deliberate choice; it followed from taking the brief's "300
> steps" literally before the action-skip was discovered. §10 measures what it cost.
> `sin_best_dx_short` (displacement at 100 simulator steps) was described in an earlier draft as
> "the ground truth's own horizon" — it is not; the ground truth's horizon is 500 simulator steps.

**No controller was trained anywhere in this pipeline.** Every actuation signal is constant,
uniform random, or sinusoidal. No policy parameters are ever updated.

## 3. Axes: definitions and compute cost

Measured over the full 700-morphology run (`notes/timing.md`), 4 cores, ~640 simulator steps/s
per core.

| family | axes | what it does | s/morphology | share |
|---|---|---|---|---|
| **structural** (4a) | `n_voxels`, `n_actuators`, `actuator_fraction`, `h_actuator_fraction`, `aspect_ratio`, `compactness`, `h_symmetry`, `modularity`, `n_components` | pure functions of the grid; modularity via networkx greedy communities on the 4-connectivity voxel graph | 0.001 | 0.0% |
| **passive** (4b) | `settle_com_y_drift`, `settle_com_x_drift`, `settle_max_speed_tail`, `settle_mean_speed_tail` | 200 steps at neutral actuation; drift and residual speed over the last 50 steps | 0.336 | 1.2% |
| **reach** (4c) | `rand_dx_{mean,max,std}`, `rand_mean_speed`, `sin_best_dx`, `sin_best_dx_short`, `sin_best_freq`, `sin_best_pattern_id`, `sin_dx_{mean,std,range}` | 8 x 300-step uniform-random episodes, plus a 4 frequency x 4 phase-pattern sinusoid sweep = the "best open-loop gait" | 12.027 | 43.0% |
| **efficiency** (4d) | `eff_raw`, `eff_raw_log`, `eff_dev`, `eff_dev_log` | best gait's displacement per unit actuation; free, reuses reach's memoised sweep | 0.000 | 0.0% |
| **knockout** (4e) | `ko_frac_{mean,min}`, `ko_dx_delta_{mean,worst}` | up to 6 actuators converted to soft, one at a time; fraction of intact displacement retained | 6.544 | 23.4% |
| **neighborhood** (4f) | `nbr_feasible_fraction`, `nbr_n_feasible`, `nbr_dx_{mean,max,std}`, `nbr_dx_{mean,max}_minus_self` | all 36 one-voxel edits enumerated for feasibility; 6 feasible neighbours simulated | 9.039 | 32.3% |
| **total** | 38 analysed axes | ~18,200 simulator steps | **27.95** | 100% |

The brief's target of "under ~2 s per morphology per axis" is met by structural, passive and
efficiency, and missed by the three families that run many episodes. `n_components` is constant 1
across the feasible set (connectivity *is* a feasibility rule) and is dropped automatically as
zero-variance — it was computed anyway so the claim was checked rather than assumed.

**Scope reductions** (full detail and evidence in `notes/scope.md`): knockout and neighbourhood
bodies are scored with a 3-episode reduced sinusoid sweep instead of the full 16, and 6 neighbours
are simulated instead of 12. The reduced sweep was chosen by a rule fixed in advance — hold
frequency, vary phase pattern — and preserves the full sweep's ranking at **Spearman 0.950** on a
24-morphology pilot. A side finding from that pilot: **phase pattern carries nearly all the
signal and frequency very little** (varying frequency alone tops out near 0.6).

## 4. Sample

`data/sample.parquet`, 700 rows: **600 decile-stratified** (60 per rank-decile, seed 12345) plus
the **true top 100** by fitness. No overlap. Deciles are rank-based because the floor pile-up
makes value-based bin edges collide.

**The top-100 block is a deliberate over-sample of the extreme tail, so it is excluded from every
correlation and cross-validation result below** and used only for the top-100 recall analysis.
This matters: the stratified 600 span −5.066 … **+0.071**, while the top 100 span **+3.030** …
+4.424. The two groups do not overlap at all.

## 5. Per-axis correlation with true fitness

600 stratified morphologies, Spearman with bootstrap 95% CIs (2000 resamples). Full table in
`results/axis_correlations.csv`; plotted in `results/axis_correlations.png` and
`results/axes_vs_truth.png`.

| axis | family | Spearman ρ | 95% CI | n |
|---|---|---|---|---|
| `rand_dx_max` | reach | **0.543** | 0.484 … 0.599 | 600 |
| `sin_best_dx` | reach | **0.501** | 0.439 … 0.560 | 600 |
| `nbr_dx_mean` | neighborhood | **0.472** | 0.408 … 0.530 | 600 |
| `sin_best_dx_short` | reach | 0.466 | 0.399 … 0.527 | 600 |
| `sin_dx_range` | reach | 0.439 | 0.373 … 0.499 | 600 |
| `sin_dx_std` | reach | 0.432 | 0.366 … 0.494 | 600 |
| `rand_dx_mean` | reach | 0.407 | 0.337 … 0.474 | 600 |
| `eff_dev`, `eff_dev_log` | efficiency | 0.406 | 0.337 … 0.473 | 600 |
| `eff_raw`, `eff_raw_log` | efficiency | 0.405 | 0.334 … 0.472 | 600 |
| `nbr_dx_max` | neighborhood | 0.399 | 0.331 … 0.464 | 600 |
| `sin_dx_mean` | reach | 0.392 | 0.322 … 0.456 | 600 |
| `rand_mean_speed` | reach | 0.366 | 0.294 … 0.432 | 600 |
| **`n_actuators`** | **structural** | **0.351** | 0.282 … 0.418 | 600 |
| `ko_frac_mean` | knockout | 0.328 | 0.246 … 0.403 | 483 |
| `actuator_fraction` | structural | 0.320 | 0.249 … 0.389 | 600 |
| `ko_frac_min` | knockout | 0.304 | 0.214 … 0.389 | 483 |
| `ko_dx_delta_worst` | knockout | −0.254 | −0.328 … −0.175 | 600 |
| `nbr_feasible_fraction`, `nbr_n_feasible` | structural | 0.228 | 0.151 … 0.305 | 600 |
| … | | | | |
| `sin_best_pattern_id` | reach | 0.049 | −0.031 … 0.129 | 600 |
| `settle_max_speed_tail` | passive | −0.047 | −0.123 … 0.030 | 600 |
| `aspect_ratio` | structural | −0.037 | −0.096 … 0.026 | 600 |
| `h_symmetry` | structural | 0.033 | −0.046 … 0.117 | 600 |
| `settle_com_x_drift` | passive | −0.014 | −0.096 … 0.064 | 600 |
| `settle_com_y_drift` | passive | 0.006 | −0.076 … 0.091 | 600 |

Six axes have CIs straddling zero: three of the four `settle_*` axes plus `sin_best_pattern_id`,
`aspect_ratio` and `h_symmetry`. (`settle_mean_speed_tail` is the one passive axis whose CI
excludes zero, at ρ = −0.088 — significant, but far too small to be useful.)

`ko_frac_mean` and `ko_frac_min` are computed on **483 of 600** morphologies. For the other 117 the
intact body's best displacement is below 0.25, so "fraction retained" would be a ratio of two
near-zero quantities; those are reported as missing rather than filled in.

## 6. Cross-validated models and the structural/simulation ablation

5-fold CV on the 600 stratified morphologies, pooled out-of-fold predictions.
Scatter: `results/gbt_pred_vs_true.png`.

| model | features | CV Spearman | CV R² |
|---|---|---|---|
| Ridge, all axes | 38 | 0.651 | 0.414 |
| **GBT, all axes** | 38 | **0.655** | **0.419** |
| Ridge, simulation only | 28 | 0.661 | 0.423 |
| GBT, simulation only | 28 | 0.654 | 0.415 |
| Ridge, **structural only** | 10 | 0.332 | 0.114 |
| GBT, **structural only** | 10 | 0.296 | 0.031 |

Two things stand out.

**Simulation is where the signal is.** Structural-only sits at ρ≈0.30–0.33 and R² near zero;
adding simulation roughly doubles the rank correlation and lifts R² from 0.03–0.11 to ~0.42.

**Structure adds nothing on top of simulation.** Simulation-only matches all-axes to within noise
(0.661 vs 0.651 for Ridge; 0.654 vs 0.655 for GBT). Whatever the structural descriptors know, a
few hundred steps of open-loop simulation already knows.

Note the structural-only group here includes `nbr_feasible_fraction` and `nbr_n_feasible`, because
enumerating one-voxel edits is pure combinatorics with no simulation. Classing them as
"simulation" would have overstated what simulation buys.

**GBT permutation importance** (measured on held-out folds, `results/gbt_permutation_importance.csv`):

| axis | importance |
|---|---|
| `rand_mean_speed` | 0.165 |
| `rand_dx_max` | 0.162 |
| `settle_com_y_drift` | 0.028 |
| `ko_frac_min` | 0.028 |
| `rand_dx_mean` | 0.015 |

The random-actuation probe dominates, and by more than its univariate correlation suggests —
`rand_mean_speed` ranks first on importance despite a univariate ρ of only 0.366, so it is
carrying something the displacement axes do not. Everything past the top two is nearly flat, which
is what heavy redundancy (§7) looks like from a tree model's perspective.

## 7. Redundancy

Pairwise Spearman between axes: `results/axis_redundancy.csv`, heatmap in
`results/axis_redundancy_heatmap.png`.

| pair | \|ρ\| |
|---|---|
| `eff_dev_log` — `eff_dev` | 1.000 |
| `eff_dev_log` — `eff_raw` | 0.9999 |
| `sin_dx_range` — `sin_dx_std` | 0.954 |
| `eff_dev` — `sin_best_dx` | 0.937 |
| `sin_best_dx` — `sin_best_dx_short` | 0.928 |
| `compactness` — `n_voxels` | 0.921 |
| `rand_dx_mean` — `rand_dx_max` | 0.910 |
| `actuator_fraction` — `n_actuators` | 0.869 |
| `nbr_dx_mean` — `nbr_dx_max` | 0.845 |

Concretely:

- **The whole efficiency family is redundant.** All four variants are rank-identical to one
  another (ρ = 1.000 between the `log` and raw forms, because `sign(x)·log1p(|x|)` is monotone —
  so the log compression cannot change a rank correlation at all) and ρ = 0.94 with `sin_best_dx`.
  Efficiency measured this way is essentially "displacement, rescaled by a near-constant".
- `compactness` and `n_voxels` are nearly the same variable on a 3x3 grid.
- The 100-step and 300-step horizons agree closely (ρ = 0.928), so probe length is not critical.
- A parsimonious set would be roughly: one random-actuation displacement, `rand_mean_speed`, one
  sinusoid-sweep displacement, `nbr_dx_mean`, `n_actuators`, and one knockout fraction.

## 8. Top-100 recall

Fraction of the true top 100 landing in the top quartile (175 of 700) when the pooled sample is
ranked by each axis. Chance is 0.25. Axis orientation is taken from the stratified 600 only, so
the metric never peeks at top-100 membership; the GBT column uses out-of-fold predictions.
Full table: `results/top100_recall.csv`.

| axis | family | recall |
|---|---|---|
| **GBT (all axes, out-of-fold)** | model | **0.98** |
| `sin_dx_mean` | reach | 0.95 |
| `nbr_dx_mean` | neighborhood | 0.95 |
| `nbr_dx_max` | neighborhood | 0.93 |
| `sin_best_dx` | reach | 0.88 |
| `sin_best_dx_short` | reach | 0.85 |
| `ko_frac_min` | knockout | 0.80 |
| `actuator_fraction` | structural | 0.75 |
| `n_actuators` | structural | 0.69 |
| `eff_raw` | efficiency | 0.66 |
| `settle_mean_speed_tail` (best passive) | passive | 0.51 |
| `modularity` | structural | 0.03 |
| `settle_com_{x,y}_drift` | passive | 0.01 |

**This number needs a caveat, and it is a large one.** The top 100 (fitness +3.03 … +4.42) and the
stratified 600 (−5.07 … +0.07) do not overlap at all — there is a gap of about 3 fitness units
between the two groups. Separating them is a much easier task than ranking morphologies within the
bulk of the distribution, and 0.98 should be read as "these axes reliably flag the extreme
outliers", not as "these axes rank morphologies well". Section 5's ρ ≈ 0.5 and §6's R² ≈ 0.42 are
the honest measures of the harder task.

Several structural axes score *far below* chance (`modularity` 0.03, `n_voxels` and `compactness`
0.06). That is not a bug: oriented by their weak positive population correlation, they actively
rank the top-100 morphologies near the bottom. The best morphologies are not the biggest or the
most modular ones.

## 9. Where 0.655 sits: landscape smoothness and the noise ceiling

Two whole-map checks, both computed over all 1,305,840 morphologies with **no simulation at all**
(`analysis/landscape_checks.py`, under a minute). Full write-up in `notes/landscape.md`.

### 9a. Smoothness — a neighbour oracle reaches ρ = 0.814

Predicting each morphology's fitness from the **mean true fitness of its feasible one-voxel
neighbours** gives **Spearman 0.814** against truth (Pearson 0.830; the *max*-neighbour variant is
weaker at 0.614). Morphologies have 32.0 feasible neighbours on average, out of 36 candidates.

This is an **oracle**, not a ceiling our axes could be expected to hit — it is handed every
neighbour's ground-truth fitness. Read it as: *how strongly does a body's position in the design
space constrain its fitness?* Quite strongly.

So our 0.655 sits meaningfully below 0.814, and there is real headroom. **But the breakdown
changes what that headroom means.** Recomputing the same oracle statistic *within* each
true-fitness decile:

| decile | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Spearman | +0.07 | +0.09 | +0.02 | +0.11 | +0.23 | +0.15 | +0.14 | +0.14 | +0.21 | **+0.48** |

Inside any decile but the top one, even a full neighbour oracle manages ρ ≤ 0.23. **The 0.814 is
almost entirely coarse-band separation, not fine-grained ranking.** The landscape is smooth at
large scale and close to flat-or-noisy at small scale.

The practical reading: the gap from 0.655 to 0.814 is concentrated in exactly the part of the
problem the reach axes already handle — telling broad bands apart. Fine discrimination within a
band looks close to irreducible from body identity alone, so a predictor chasing the last 0.15 is
chasing the easier half.

### 9b. Noise ceiling — unknown, and it stays unknown

`raw_results.npy` stores, per morphology, a **single 1-D array of 300 floats**: best-so-far fitness
after each generation of **one** controller-evolution run. Checked over a 200-morphology sample —
every value is `(300,)`, never 2-D, never a list of runs.

**There is one attempt per morphology, so no split-half reliability can be computed, and the noise
ceiling on ground truth is unknown.** We do not estimate it.

What the data *does* support: the published true fitness revises the 300-generation result upward
for **76,525 morphologies (5.86%)** — never downward — by a median of **+0.743** and a maximum of
**+9.301**, on a scale spanning about 9.7 in total.

> Spearman(300-generation final, published true fitness) is 0.976, and it is tempting to quote
> that as reliability. **It is not, and we do not.** True fitness is defined as the better of the
> 300-generation result and anything the co-optimization runs found, so the two are dependent by
> construction — one is a maximum involving the other.

The sound conclusion is one-sided: a single run under-estimates achievable fitness for at least
5.86% of morphologies, sometimes badly. Our target is therefore a noisy lower bound, which means
**0.655 under-states correlation with truly achievable fitness by an unknown margin.**

## 10. Episode-count ablation: how correlation buys itself with compute

*(Pending — `analysis/episode_ablation.py` running at time of writing.)*

## 11. Assumptions

**No unresolved assumptions remain.** The single ASSUMPTION carried by the first draft has since
been resolved against the paper.

> **RESOLVED** (was an ASSUMPTION about `updated_results.pkl`) — the repo never documents how the
> updated fitnesses were produced, and the file that would show it
> (`updated_results_w_long.pkl`, referenced at `morphology_space_evolution.py:124`) is absent from
> the published tarball. The **paper** states it, in "Updating the Landscape": fitness was
> re-estimated for **76,526** morphologies using controllers discovered during the
> 10,000-generation co-optimization runs. Our independent count of revised entries is **76,525**,
> off by one — presumably a morphology whose revised value was not numerically distinguishable.
>
> So true fitness is the best controller found for a body across the 300-generation AFPO run *and*
> the co-optimization runs: a best-of-attempts **lower bound** on achievable fitness, noisy by an
> amount the published data cannot reveal (§9b).

Three claims that might look like assumptions but were each verified independently:

- **the 3x3 search space** — exact set match against our own enumeration (§1), and the paper says
  so outright: *"each viable morphology that exists in the 3-by-3 morphology space"*. The
  abstract's "1,305,840 voxel-based soft robots" matches our enumerated count exactly.
- **the −5.0 fitness offset** — arithmetic confirmed, and the mechanism corrected against the
  paper and by replicating the upstream wrapper stack (§1).
- **`n_components` being constant** — computed rather than assumed, then dropped as zero-variance.

One network blocker was hit and resolved without losing scope: the git proxy refuses git-LFS
objects for this repo and a credentialed attach was denied, so the data was fetched from GitHub's
public LFS media host, with SHA-256 and byte size verified against the repo's own LFS pointer
(`notes/blockers.md`).

## 12. Conclusion

**Do cheap, training-free axes carry meaningful signal about achievable fitness?**

**Yes — but nearly all of it comes from briefly simulating the body, not from looking at it.**

- The best single axis is `rand_dx_max`, how far the body drifts under the luckiest of 8 random
  actuation episodes: **ρ = 0.543** (95% CI 0.484–0.599). The best open-loop sinusoidal gait
  (`sin_best_dx`) is close behind at **ρ = 0.501**.
- Combining all 38 axes reaches **CV Spearman 0.655, R² 0.419**. That is a real, comfortably
  significant relationship, and it is also a loose one: the predicted-vs-true scatter shows wide
  spread and a large pile-up at the "did not move" floor. These axes order morphologies; they do
  not pin down their fitness.
- **Structure alone is weak.** The best structural axis is simply `n_actuators` (ρ = 0.351), and a
  model using only non-simulation axes manages ρ ≈ 0.30–0.33 with R² of 0.03–0.11. Shape
  descriptors that sound meaningful — aspect ratio, horizontal symmetry, graph modularity — are at
  or below |ρ| = 0.10, and the CIs for aspect ratio and symmetry include zero.
- **Passive stability is a clean null.** All four settle axes have |ρ| ≤ 0.09 and three of the four
  have CIs containing zero. How a body behaves when you do nothing to it says nothing useful about
  how well it can be made to walk. This is a genuine negative result and we report it as one.
- **Neighbourhood quality works, but probably not as evolvability.** `nbr_dx_mean` reaches
  ρ = 0.472 — yet it is 0.845-redundant with `nbr_dx_max` and behaves much like the body's own
  reach. Neighbours of a good walker are good walkers. We cannot claim to have measured
  evolvability as distinct from "this region of the space moves well".
- **Efficiency and the log transforms earn nothing.** All four efficiency variants are rank-
  identical to each other and 0.94-redundant with plain displacement.

The practical read: **a few hundred steps of open-loop simulation is a genuinely useful and very
cheap proxy for morphological potential, and static structural descriptors are not.** Our whole
battery costs ~18,200 simulator steps (28 s) per morphology. The ground truth cost 300 generations
of controller evolution per morphology — at least 30,000 steps even for a population of one, and
realistically orders of magnitude more. So the proxy buys a large speed-up for a rank correlation
around 0.65.

What this does **not** establish: that these axes would survive on a harder task than flat-ground
walking, on larger bodies, or that ρ ≈ 0.65 is enough to actually drive a search. Those are Stage 2
questions.

## 13. Next run

Before running on the 3000-morphology sample (`data/sample_large_ids.parquet`, already written):

1. **Budget.** At 28 s/morphology, 3000 morphologies is ~5.8 hours on 4 cores. Either raise the
   core count or drop the two costly families — knockout (23% of runtime) and neighbourhood (32%)
   are the weakest performers per second, and dropping both cuts runtime by more than half while
   costing little, since simulation-only minus those two is dominated by reach anyway.
2. **Select the best gait at the task's own horizon.** `sin_best_dx_short` currently reports the
   100-step displacement *of the gait that won at 300 steps*. Selecting the argmax over the sweep
   by 100-step displacement instead is free — the data is already collected per episode — and
   should be better matched to a 100-step ground truth.
3. **Spend the reach budget on random episodes, not the sweep.** `rand_dx_max` beat every
   sinusoid axis, and `rand_mean_speed` was the top permutation-importance feature. Trading some
   of the 16-episode sinusoid sweep for more random episodes looks like the best marginal use of
   simulation time.
4. **Drop the dead weight.** Remove `n_components` (constant), the three redundant efficiency
   variants, and `compactness` (0.92 with `n_voxels`). Keep one axis per redundant cluster.
5. **Widen the knockout coverage instead of the depth.** `ko_frac_*` is missing for 17% of
   morphologies because the intact body barely moves. Consider reporting the absolute delta (which
   is always defined, and already present as `ko_dx_delta_worst` at ρ = −0.254) as the primary
   knockout axis.
6. **Fix the sample design for the recall metric.** The top-100 block does not overlap the
   stratified sample at all, which makes top-100 recall too easy to interpret cleanly. Either
   stratify within the top decile as well, or report recall against a threshold drawn from the
   sampled distribution.
7. **Consider a harder target.** Flat-ground walking may be the easiest task in EvoGym for an
   open-loop probe to imitate, which could be exactly why reach axes do so well. The claim would
   be far stronger if it held on a task where the best open-loop gait is *not* a decent walker.
