# Seeds

All randomness in this project is seeded. Seeds are recorded here as they are introduced.

| Constant | Value | Where | Used for |
|---|---|---|---|
| `SMOKE_SEED` | 20240517 | `tests/test_evogym_smoke.py` | random valid body for the EvoGym smoke test |

(Upstream, for reference: the ground-truth runs seeded `random`/`numpy`/`torch` with the
morphology ID itself, and seeded the env with 17 — see `notes/dataset.md`.)
