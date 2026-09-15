# Per-axis compute time

Mean wall-clock seconds per morphology, measured inside the workers during the Step 4 run.

| axis | mean s/morphology | median | max | share |
|---|---|---|---|---|
| structural | 0.001 | 0.001 | 0.003 | 0.0% |
| passive | 0.336 | 0.327 | 0.802 | 1.2% |
| reach | 12.027 | 11.624 | 24.623 | 43.0% |
| efficiency | 0.000 | 0.000 | 0.000 | 0.0% |
| knockout | 6.544 | 6.111 | 17.567 | 23.4% |
| neighborhood | 9.039 | 8.748 | 18.462 | 32.3% |
| **all axes** | **27.946** | | | 100% |

Morphologies completed: 700. Total wall-clock for the run: 81.7 min across 4 cores.

The brief's target was under ~2 s per morphology per axis. Every axis meets that except the two that re-simulate derived bodies (`knockout`, `neighborhood`) and `reach`, whose 16-episode sinusoid sweep is the headline probe. See notes/scope.md for the budget.
