# FreshSky-Safe strict v3 evidence and visual contract

This directory contains the seed-level evidence used by the revised TMC
manuscript. All comparisons use common exogenous random streams within a cell.
The primary baseline design contains 11 policies x 3 feedback delays x 20
seeds (660 runs).

## Evidence families

- `tmc_extended_baselines_*`: 11 policies, delays 2/4/8, 20 seeds per cell.
- `tmc_factorial_ablation_*`: full 2 x 2 x 2 mechanism design, 20 seeds per
  cell.
- `tmc_qcap_service_*`: nine queue caps, 20 seeds per point.
- `tmc_delayed_shift_*`: four policies, 20 paired seeds.
- `derived/`: figure-facing effects, rank stability, seed-preserving
  multi-outcome contrasts, queue-pressure diagnostics, runtime grids, and
  per-UAV long-form data generated from the seed files.

The calibration, measured-replay, and 1000-repetition runtime files are stored
one directory above after synchronization to `experiments/results/`.

## Headline findings

- FreshSky W-AoI is 436.084, 445.082, and 454.274 at pending-feedback delays
  2, 4, and 8.
- Zhao'25 MMSE-MW-Safe is unresolved against FreshSky at delay 2, then lowers
  W-AoI by 8.001 slots [1.00, 15.00] and 15.307 slots [9.48, 21.14] at
  delays 4 and 8. Its fixed Theorem-7 beta, posterior-predictive success
  adaptation, and common shield are documented in
  `experiments/zhao25_baseline_fidelity.md`.
- Wang'26 PORMAB-Safe and Tripathi'24 Whittle-Safe are unresolved against
  FreshSky at delays 2 and 4, and outperform it by 4.00% and 3.88% at delay 8.
- Relative to the different-contract Zhu'26 aggregate-DPP point, FreshSky pays a
  7.43--8.28% W-AoI cost, reduces maximum individual power by
  28.28--29.65%, and changes certification from 0/20 to 20/20.
- The expected-age factorial effect is -174.652 slots. The shield adds 6.961
  slots while saving 0.00900 mW in maximum individual power.
- Raising the queue cap from 4 to 48 reduces endpoint W-AoI by 5.21% and
  candidate rejection from 34.17% to 0.64%, but the best observed mean W-AoI
  occurs at cap 36 and starvation p95 worsens.
- Under the tested abrupt dwell shift, the static control beats FreshSky
  post-shift by 14.437 slots (2.75%).

## v3 figure evidence

- Fig. 3 retains both 12-cell calibration maps, a four-outcome measured-trace
  matrix, and all 120 matched measured seed pairs.
- Fig. 4 combines the complete 11-policy x 3-delay effect matrix, d=4 rank
  stability, the broken-axis contract plane, and paired pre/post shift
  differences.
- Fig. 5 exposes all eight raw factorial cells, all 35 outcome-by-effect
  summaries, and the 80 paired shield trade-offs.
- Fig. 6 uses the seed as the inferential unit for its nine-point operating
  path, four pressure diagnostics, and 90 raw tail-matrix cells.
- Fig. 7 displays all 54 measured runtime points plus the component and memory
  envelopes; it does not extrapolate beyond the measured grid.

## Interpretation boundary

The strict rerun does not support universal freshness dominance. The
deterministic result is the per-UAV finite-window energy certificate under the
stated service-before-arrival queue and queue-cap admission test. Freshness,
fairness, transfer, and runtime results remain empirical.

## Reproduce

From the project root:

```powershell
python experiments\run_tmc_evidence.py
```

To validate completed outputs and write hashes without rerunning:

```powershell
python experiments\run_tmc_evidence.py --manifest-only
```

The Zhao formula and shield checks can be run independently with:

```powershell
python experiments\test_zhao25_specialization.py
```
