# IoTJ inference evidence summary

All prediction comparisons use the same round-robin probe decisions, hidden-channel realizations, and ACK draws. The seed, not an individual ACK, is the inferential unit.

## Stationary common-probe calibration

| NLoS dwell | delay | Bayesian Brier skill vs static geometry | Bayesian Brier gap to known-law filter |
|---:|---:|---:|---:|
| 2 | 2 | -0.82% ± 0.39% | 1.92% ± 0.09% |
| 2 | 4 | -0.63% ± 0.36% | 1.62% ± 0.09% |
| 2 | 8 | -0.54% ± 0.35% | 1.53% ± 0.09% |
| 8 | 2 | 3.31% ± 0.58% | 2.05% ± 0.07% |
| 8 | 4 | -1.46% ± 0.16% | 2.48% ± 0.17% |
| 8 | 8 | -2.59% ± 0.20% | 2.80% ± 0.22% |
| 14 | 2 | 12.39% ± 1.08% | 2.18% ± 0.10% |
| 14 | 4 | 2.47% ± 0.62% | 2.77% ± 0.17% |
| 14 | 8 | -1.11% ± 0.34% | 3.27% ± 0.25% |

## Abrupt-shift late-phase result

At delay 4, the Bayesian ensemble changes Brier score by 2.37% ± 0.57% relative to the filter that remains fixed at the pre-shift transition law.

The open-loop inference experiment diagnoses perception/reasoning only. It does not replace the closed-loop scheduler comparison and does not imply universal W-AoI dominance.
