# The ground-truth dataset

Source: `mertan-a/morphology-fitness-landscape` (Mertan & Cheney, *"Evolutionary Brain-Body
Co-Optimization Consistently Fails to Select for Morphological Potential"*, arXiv 2508.17464).

Everything below was read off the upstream code and verified against the data. Nothing here is
assumed unless it is explicitly marked **ASSUMPTION**.

---

## 1. The search space is 3x3, not 5x5

**This is the single most important correction to the project brief.** The brief describes "every
feasible 5x5 EvoGym morphology". The published map is over a **3x3** bounding box.

Evidence, in order of strength:

1. The published IDs run from **93 to 1,953,124**, and `5**9 - 1 = 1953124` exactly. A 5x5 space
   would use IDs up to `5**25 - 1 = 298023223876953124`.
2. I enumerated all `5**9 = 1,953,125` 3x3 grids and applied the upstream feasibility rules. That
   yields **1,305,840** feasible grids — exactly the number of keys in `updated_results.pkl` — and
   the two sets are an **exact set match** (`set(enumerated) == set(published)`, zero in either
   difference). Reproduced as a test in `tests/test_id_mapping.py`.
3. A 25-cell exhaustive map is not computationally plausible: 5^25 ≈ 3x10^17 candidates.

**Decision: the whole project runs on the 3x3 space**, because that is the space the ground truth
covers, and correlating cheap axes against ground truth is the entire point. `BOUNDING_BOX` in
`axes/common.py` is `(3, 3)`. The smoke test still exercises a 5x5 body as well, to show EvoGym
itself is not the constraint.

Consequences for later steps: the one-voxel neighbourhood (Step 4f) has at most 9x4 = 36 candidates
rather than 25x4 = 100, and a morphology has at most 9 actuators, so the Step 4e knockout budget of
6 is a real cap rather than a formality.

## 2. ID -> grid mapping

From `search_space.py` (`integer_idx_to_ndarray` / `ndarray_to_integer_idx`):

> The ID is the integer whose **base-5** representation, left-padded with zeros to 9 digits, lists
> the voxel types in **row-major** order.

Voxel type codes are EvoGym's:

| code | meaning |
|---|---|
| 0 | empty |
| 1 | rigid |
| 2 | soft |
| 3 | horizontal actuator |
| 4 | vertical actuator |

Worked examples (all are published IDs, all verified feasible):

```
id 93       -> [[0 0 0]      id 1953124 -> [[4 4 4]      id 1514953 -> [[3 4 1]
                [0 0 0]                     [4 4 4]       (the global   [4 3 4]
                [3 3 3]]                    [4 4 4]]       optimum)     [3 0 3]]
```

`93 = 3*5^2 + 3*5^1 + 3*5^0` -> digits `000000333`. It is the smallest feasible ID: a bottom row of
three horizontal actuators.

## 3. Feasibility rules

A grid is in the search space iff all three hold (`search_space.create_seach_space`):

1. `evogym.is_connected(grid)` — the non-empty voxels form one connected component;
2. `np.sum(grid > 0) >= 3` — at least 3 non-empty voxels;
3. `np.sum(grid == 3) + np.sum(grid == 4) >= 3` — at least 3 actuator voxels.

Re-implemented in `axes.common.is_feasible` and cross-checked against upstream in
`tests/test_id_mapping.py`.

## 4. The task and environment

From `simulator.py` and `main.py` upstream:

| setting | value |
|---|---|
| task | **`Walker-v0`** (EvoGym `WalkingFlat`, flat-ground locomotion) — the only choice `main.py` allows |
| connections | `evogym.get_full_connectivity(body)` |
| wrappers | `RecordEpisodeStatistics`, then `RewardShapingWrapper` (**-0.05 reward per step**) |
| seeds | `env.seed(17)`, `action_space.seed(17)`, `observation_space.seed(17)` |
| episode length | **100 steps** (see §5 — this is *not* the 500 the code appears to ask for) |
| controller | a fixed-topology MLP, weights evolved by AFPO for 300 generations; body held fixed |

EvoGym specifics confirmed locally (simulator v2.2.5): action space is `Box(0.6, 1.6)` per
actuator — a target volume ratio, with **1.0 neutral** — `VOXEL_SIZE = 0.1`, and a 3x3 robot is
spawned spanning x in [0.1, 0.4], i.e. COM x = 0.25.

## 5. What the fitness number means

`WalkingFlat.step` gives `reward = com_x(t+1) - com_x(t)`, so an episode's undiscounted return is
simply the robot's **net centre-of-mass x-displacement in world units**. On top of that:

- `RewardShapingWrapper` subtracts **0.05 per step**;
- reaching `com_x > 9.9` ends the episode with a **+1.0** bonus;
- an unstable simulation ends the episode with **-3.0**.

So **fitness = (net COM x-displacement) - 0.05 x (number of steps)**.

### Episode length is 100 steps, established empirically

The upstream code *looks* like it asks for 500 steps, but `simulator.make_env` writes
`env.env.env.env.env._max_episode_steps = 500`, which walks past the `TimeLimit` wrapper and sets a
dead attribute on an inner env. So the registered default governs. The data settles which it was:

| | measured / implied |
|---|---|
| ground-truth generation-1 fitness | min -5.7735, max -4.6480, **mean -4.9883** |
| implied net displacement if 100 steps (offset -5.0) | min -0.774, max +0.352, **mean +0.012** |
| implied net displacement if 500 steps (offset -25.0) | min +19.23, max +20.35, **mean +20.01** |
| my measured displacement, random actions, 100 steps, 30 morphologies | min -0.106, max +0.261, **mean +0.010** |
| my measured displacement, random actions, 500 steps, 30 morphologies | min -0.873, max +1.667, mean -0.048 |

Generation-1 controllers are random networks, so their net displacement should be near zero. Under
a 100-step episode the implied mean is **+0.012** against my measured **+0.010** — essentially
exact. Under 500 steps it would have to be **+20**, which is over an order of magnitude beyond
anything reachable even at 500 steps. **Episode length is 100 steps.**

Two independent consistency checks pass:

- The best morphology's fitness of 4.4237 implies a displacement of 9.4237. Reaching the goal needs
  9.65 (from COM x = 0.25 to 9.9) and would have paid a +1.0 bonus and ended the episode early, so
  every goal-reaching robot would score at least ~5.65. **No morphology scores above 4.4237**, so
  none ever reached the goal — consistent, and it explains why the fitness distribution stops just
  short of the goal threshold.
- No trajectory anywhere shows the -3.0 signature of an unstable simulation.

**Interpretation for the report:** a fitness of -5.0 means "did not move"; every point above -5.0 is
one world unit (10 voxel widths) of net forward travel. The observed range is -5.2718 to +4.4237,
i.e. from slight backward drift to ~9.4 units of travel in 100 steps.

## 6. The two results files

| file | contents |
|---|---|
| `raw_results.npy` | dict `{id: np.ndarray of 300 float64}` — best-so-far fitness per generation |
| `updated_results.pkl` | dict `{id: float}` — the estimated **true fitness** |

Both have exactly the same 1,305,840 keys. Verified over the full set:

- Every one of the 1,305,840 trajectories is **monotone non-decreasing** (0 violations), confirming
  they record best-so-far, so `traj[-1] == max(traj)` everywhere.
- `updated_results[id]` differs from `raw_results[id][-1]` for **76,525 IDs (5.86%)**, and in those
  cases it is **always larger** (0 cases smaller), by up to **9.30**.

So the "updated" map takes the 300-generation result and revises a minority of morphologies upward.

> **ASSUMPTION:** the upstream README says `updated_results.pkl` holds "the estimated true fitness"
> but does not describe the update procedure, and the file that would show it
> (`updated_results_w_long.pkl`, referenced in `morphology_space_evolution.py:124`) is **not in the
> published tarball**. Given that revisions are strictly upward and affect only ~6% of
> morphologies, the natural reading is that promising morphologies were re-evaluated with longer or
> repeated controller-evolution runs and the best result kept. **We take
> `updated_results.pkl` at face value as the ground-truth "true fitness" target**, which is what the
> README instructs. This does not affect our method: we only need a fixed per-morphology scalar to
> correlate against. It does mean true fitness is a *best-of-attempts* estimate and therefore
> slightly optimistic and noisy, which we note when interpreting correlation ceilings.

## 7. Distribution of true fitness

Over all 1,305,840 morphologies:

| statistic | value |
|---|---|
| min | -5.2718 |
| 1st pct | -5.0228 |
| 25th pct | -4.9251 |
| median | -4.2694 |
| 75th pct | -3.0524 |
| 99th pct | -0.4965 |
| max | +4.4237 |
| mean | -3.8793 |

The distribution is heavily **left-concentrated**: a quarter of all morphologies sit within 0.08 of
"did not move at all" (-5.0), and the median moves only ~0.73 world units in 100 steps. Good
morphologies are rare. This matters for Step 3 — a uniform random sample would be almost entirely
near-immobile robots, which is why the brief's decile-stratified sample is the right call — and for
Step 5, where large ties near the floor will attenuate rank correlations.

## 8. Derived file committed to this repo

`data/ground_truth.parquet` (32 MB, zstd) — one row per morphology, columns:

| column | meaning |
|---|---|
| `id` | morphology ID (int64) |
| `true_fitness` | value from `updated_results.pkl` — the target we predict |
| `raw_final` | `raw_results[id][-1]`, the 300-generation best-so-far |
| `raw_gen1` | `raw_results[id][0]`, generation-1 fitness (used for the §5 analysis) |

This makes every later step independent of the 3.2 GB raw file, which is gitignored.
