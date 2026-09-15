# Scope reductions

The brief caps any single step at roughly 90 minutes of compute. Measured EvoGym throughput on
this sandbox is about **640 simulator steps per second per core**, on 4 cores, which makes the
Step 4 probe budget the binding constraint. Every reduction made to fit it is listed here, with
the evidence behind it.

## Measured costs

| probe | episodes x steps | seconds/morphology |
|---|---|---|
| structural (4a) | none | ~0.00 |
| passive settle (4b) | 1 x 200 | 0.31 |
| random reach (4c) | 8 x 300 | 3.82 |
| sinusoid sweep (4c, shared with 4d/4e/4f) | 16 x 300 | 7.61 |

Running the **full** 16-episode sweep for each knockout and each neighbour, as a literal reading
of the brief implies, would add `6 x 7.61 + 12 x 7.61 = 137 s` per morphology — about 27 hours of
wall clock for 700 morphologies on 4 cores. That is the reduction driver.

## Reduction 1 — a reduced sinusoid sweep for the derived bodies (4e, 4f)

Knockout and neighbourhood bodies are scored with a 3-episode reduced sweep instead of the full
16. The reduced set was chosen by a **rule fixed in advance** — hold the frequency at its most
productive value and sweep the symmetry-breaking phase patterns — not by taking the argmax over
all subsets, which would overfit the pilot:

```
REDUCED_SWEEP = (0.10, "left_right"), (0.10, "top_bottom"), (0.10, "random")
```

Validated on a pilot of 24 morphologies spanning the whole fitness range (`/tmp/pilot_sweep.csv`
during the run; the numbers are reproduced here):

| candidate reduced set | episodes | Spearman vs full 16-episode sweep | mean shortfall in dx |
|---|---|---|---|
| all 16 (reference) | 16 | 1.000 | 0.000 |
| **freq 0.10, all 4 phase patterns** | 4 | **0.980** | 0.177 |
| **freq 0.10, 3 symmetry-breaking patterns (chosen)** | 3 | **0.950** | 0.216 |
| freq 0.10, 2 patterns (left_right, random) | 2 | 0.897 | 0.393 |
| freq 0.05 + 0.10, pattern = random only | 2 | 0.607 | 0.641 |
| freq 0.05/0.10/0.20, pattern = random only | 3 | 0.583 | 0.580 |

The clear finding: **phase pattern carries nearly all the signal, frequency very little.** Varying
frequency at a fixed phase pattern tops out near 0.6, while fixing frequency and varying phase
reaches 0.95-0.98. The 3-episode set preserves the full sweep's ranking at rho = 0.950 for a
fifth of the cost.

The 16 winning (frequency, pattern) combinations were spread across all 16 cells in the pilot with
no single dominant winner, which is why the *full* sweep is kept for the headline `reach` axis
(4c) and only the derived bodies use the reduced one.

## Reduction 2 — 6 neighbours simulated instead of 12 (4f)

`MAX_NEIGHBORS_SIMULATED = 6` rather than the brief's 12. The *feasible fraction* is still
computed over **all** 36 one-voxel changes — that part is free — so only the neighbour-quality
statistics (`nbr_dx_mean`, `nbr_dx_max`, `nbr_dx_std`) rest on 6 samples. Those statistics are
correspondingly noisier, which the report notes when interpreting them.

## What was NOT reduced

- The sample is the full 700 morphologies the brief asks for (600 decile-stratified + top 100).
- All 8 random-actuation episodes (4c) at the specified 300 steps.
- The full 4 x 4 sinusoid sweep for the `reach` axis itself.
- All 6 knockouts (4e) — 6 is already at or above the actuator count for most 3x3 bodies.
- Episode length stays at the specified 300 steps. Net displacement over the ground truth's own
  100-step horizon is recorded alongside it for free as `sin_best_dx_short`.

## Resulting budget

About **25 seconds per morphology**, roughly **73 minutes** of wall clock for 700 morphologies on
4 cores — inside the 90-minute cap. Actual per-axis timings are in `notes/timing.md`.
