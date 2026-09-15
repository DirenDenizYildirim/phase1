# Morphological Potential Vector — Stage 1 (axis validation)

## Project description

This project tests whether cheap, **training-free** "potential axes" computed from a robot
morphology alone predict that morphology's ground-truth achievable fitness. The ground truth is
Mertan & Cheney's exhaustive morphology-fitness map (arXiv 2508.17464, repo
`mertan-a/morphology-fitness-landscape`): for every feasible 5x5 EvoGym morphology, a controller
was evolved for 300 generations on flat-ground locomotion (`Walker-v0`) and an estimated "true
fitness" recorded. We sample morphologies from that map, compute a battery of cheap structural and
short-simulation axes (no controller training of any kind), and measure how much signal those axes
carry about achievable fitness — via rank correlation, cross-validated regression, and top-100
recall. A null result (no axis correlates) is a valid outcome and is reported as such.

## How to work in this environment

- This runs in a cloud sandbox that can be reclaimed if idle. Assume the VM may disappear
  at any point. Every step must be resumable from what is committed in git.
- Commit and push after every step (and after any expensive computation) with a clear
  message. Never leave more than ~20 minutes of work uncommitted.
- Make every step idempotent: check for existing outputs first and skip work already done.
  Write intermediate results incrementally (append to parquet/CSV every N morphologies),
  not all at once at the end.
- Do not wait for me. Where a human check would normally be needed, write your findings
  and any assumptions to a file, commit, and continue with the most defensible assumption.
  Mark such assumptions clearly with "ASSUMPTION:" in both the file and the commit message.
- Network access is limited to package registries and allowlisted domains. github.com and
  PyPI should work. If any download is blocked, do not work around it; record the exact
  domain and error in ./notes/blockers.md, commit, and continue with whatever is possible.
- Keep total compute modest. If any single step looks like it will exceed ~90 minutes,
  reduce its scope (fewer morphologies, fewer probes), document the reduction in
  ./notes/scope.md, and continue.

## Rules

- Never train a controller. If you find yourself doing RL, ES, or gradient steps on a
  policy, stop.
- Fixed seeds everywhere; record them in ./notes/seeds.md.
- Never fabricate or interpolate missing results. Missing is missing.
- Prefer small, boring, testable code over clever code. Every axis has a unit test.
- Keep all derived data files under 50 MB so they can be committed.

## Environment quick reference

- **Python 3.10 is required** (`/usr/bin/python3.10`). The `evogym==2.0.0` wheel is built for
  `>=3.7,<3.11`; on 3.11 there is no installable distribution.
- venv lives at `./.venv`. Recreate with:
  `python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt`
- `setuptools` must stay `<81` — `evogym.envs.base` imports `pkg_resources`.
- Ground-truth data is git-LFS in the upstream repo and the session git proxy refuses LFS.
  See `notes/setup.md` for the working fetch route and `notes/blockers.md` for what failed.
- Run tests with `.venv/bin/python -m pytest tests/ -q`.

## Layout

- `axes/` — one module per axis family; each exposes a pure `grid -> dict[str, float]` function.
- `tests/` — pytest unit tests, one per axis plus dataset/mapping tests.
- `analysis/run_analysis.py` — rerunnable analysis over `data/axes.parquet` + `data/sample.parquet`.
- `notes/` — setup.md, dataset.md, seeds.md, timing.md, scope.md, blockers.md.
- `results/` — plots and `REPORT.md`.
- `data/` — committed derived parquet files; `data/mc-landscape/` is the raw upstream clone
  (gitignored).
