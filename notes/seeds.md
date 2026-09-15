# Seeds

All randomness in this project is seeded. Seeds are recorded here as they are introduced.

| Constant | Value | Where | Used for |
|---|---|---|---|
| `SMOKE_SEED` | 20240517 | `tests/test_evogym_smoke.py` | random valid body for the EvoGym smoke test |

(Upstream, for reference: the ground-truth runs seeded `random`/`numpy`/`torch` with the
morphology ID itself, and seeded the env with 17 — see `notes/dataset.md`.)

## Step 3 — sampling (`analysis/make_sample.py`)

| Constant | Value | Used for |
|---|---|---|
| `SEED_SAMPLE` | 12345 | the 600-morphology decile-stratified first-run sample |
| `SEED_SAMPLE_LARGE` | 54321 | the 3000-morphology sample for a later run |

## Step 4 — axis probes (`axes/sim.py`, `axes/knockout.py`, `axes/neighborhood.py`)

| Constant | Value | Used for |
|---|---|---|
| `PROBE_SEED` | 17 | env seed for every probe episode — the same value the ground-truth runs used |
| random-episode streams | `PROBE_SEED * 1000 + episode_index` | the 8 uniform-random reach episodes |
| `RANDOM_PHASE_SEED` | 90210 | the fixed per-body phases of the "random" sinusoid phase pattern |
| `KNOCKOUT_SEED` | 4242 | which actuators are knocked out, and in what order |
| `NEIGHBOR_SEED` | 777 | which feasible one-voxel neighbours get simulated |

Every probe is deterministic: re-running `analysis/compute_axes.py` on the same morphology
reproduces its axis values exactly (asserted in `tests/test_sim_axes.py`).
