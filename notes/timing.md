# Per-axis compute time

Mean wall-clock seconds per morphology, measured inside the workers during the Step 4 run.

| axis | mean s/morphology | median | max | share |
|---|---|---|---|---|
| structural | 0.002 | 0.002 | 0.004 | 0.0% |
| passive | 0.420 | 0.422 | 0.430 | 1.8% |
| reach | 11.399 | 11.333 | 11.751 | 47.5% |
| efficiency | 0.000 | 0.000 | 0.000 | 0.0% |
| knockout | 4.172 | 4.155 | 4.764 | 17.4% |
| neighborhood | 8.026 | 8.048 | 8.448 | 33.4% |
| **all axes** | **24.020** | | | 100% |

Morphologies completed: 4. Total wall-clock for the run: 0.4 min across 4 cores.

The brief's target was under ~2 s per morphology per axis. Every axis meets that except the two that re-simulate derived bodies (`knockout`, `neighborhood`) and `reach`, whose 16-episode sinusoid sweep is the headline probe. See notes/scope.md for the budget.
