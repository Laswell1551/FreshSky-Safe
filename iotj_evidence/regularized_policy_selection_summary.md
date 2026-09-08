# Geometry-anchored Bayesian policy selection

Validation selected alpha=0.1. The static endpoint alpha=0 was eligible, so the protocol could reject learned decision beliefs.

## Validation (seeds 0--9, all nine cells)

Each seed is first averaged over the nine dwell-delay cells; the confidence interval is then formed across seeds.

| alpha | mean W-AoI skill vs alpha=0 | 95% CI | seeds |
|---:|---:|---:|---:|
| 0 | 0.000% | ±0.000% | 10 |
| 0.1 | 0.348% | ±0.565% | 10 |
| 0.25 | 0.049% | ±0.370% | 10 |
| 0.5 | 0.068% | ±0.474% | 10 |
| 0.75 | 0.176% | ±0.545% | 10 |
| 1 | -0.885% | ±0.588% | 10 |

## Frozen test (seeds 20--39)

Across-grid inference first averages the nine cells within each seed. The selected policy changes W-AoI by 0.08% ± 0.32% relative to static geometry and improves it by 1.40% ± 0.49% relative to unregularized Bayesian decisions.

| dwell | delay | skill vs static | gain vs unregularized Bayesian |
|---:|---:|---:|---:|
| 2 | 2 | 0.65% ± 1.28% | 0.98% ± 0.97% |
| 2 | 4 | -0.45% ± 0.86% | 0.14% ± 1.05% |
| 2 | 8 | 0.88% ± 1.07% | 0.11% ± 1.03% |
| 8 | 2 | 0.05% ± 1.18% | 2.31% ± 1.51% |
| 8 | 4 | -0.51% ± 1.45% | 2.26% ± 1.49% |
| 8 | 8 | 0.17% ± 1.14% | 1.40% ± 1.21% |
| 14 | 2 | -0.26% ± 1.02% | 0.34% ± 1.18% |
| 14 | 4 | 0.14% ± 1.25% | 2.61% ± 1.40% |
| 14 | 8 | 0.01% ± 1.07% | 2.42% ± 1.45% |
