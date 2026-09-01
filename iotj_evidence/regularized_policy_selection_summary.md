# Geometry-anchored Bayesian policy selection

Validation selected alpha=0.1. The static endpoint alpha=0 was eligible, so the protocol could reject learned decision beliefs.

## Validation (seeds 0--9, all nine cells)

Each seed is first averaged over the nine dwell-delay cells; the confidence interval is then formed across seeds.

| alpha | mean W-AoI skill vs alpha=0 | 95% CI | seeds |
|---:|---:|---:|---:|
| 0 | 0.000% | ±0.000% | 10 |
| 0.1 | 0.372% | ±0.463% | 10 |
| 0.25 | 0.164% | ±0.328% | 10 |
| 0.5 | 0.218% | ±0.519% | 10 |
| 0.75 | 0.195% | ±0.668% | 10 |
| 1 | -0.708% | ±0.528% | 10 |

## Frozen test (seeds 20--39)

Across-grid inference first averages the nine cells within each seed. The selected policy changes W-AoI by 0.02% ± 0.31% relative to static geometry and improves it by 1.23% ± 0.37% relative to unregularized Bayesian decisions.

| dwell | delay | skill vs static | gain vs unregularized Bayesian |
|---:|---:|---:|---:|
| 2 | 2 | 0.65% ± 1.28% | 0.98% ± 0.97% |
| 2 | 4 | -0.45% ± 0.86% | 0.14% ± 1.05% |
| 2 | 8 | 0.88% ± 1.07% | 0.10% ± 1.03% |
| 8 | 2 | 0.30% ± 1.08% | 2.07% ± 1.18% |
| 8 | 4 | -0.84% ± 1.37% | 1.81% ± 1.40% |
| 8 | 8 | -0.08% ± 1.37% | 1.25% ± 1.27% |
| 14 | 2 | -0.16% ± 1.06% | 0.17% ± 1.38% |
| 14 | 4 | -0.01% ± 1.11% | 2.36% ± 1.19% |
| 14 | 8 | -0.13% ± 1.04% | 2.19% ± 1.12% |
