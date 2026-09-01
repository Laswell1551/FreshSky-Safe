# FreshSky-Safe IoTJ Reproducibility Artifact

This release-ready package accompanies:

> **FreshSky-Safe: Certified Freshness Scheduling for Aerial IoT over Hidden
> Channels with Delayed Feedback**

It contains executable source, fixed configurations, seed-level outputs,
figure generators, a measured air-ground channel replay, and machine-readable
claim checks. The public repository is
<https://github.com/Laswell1551/FreshSky-Safe>. An archival DOI will be added
if a release is deposited with a long-term archive.

## Artifact scope

The artifact implements and validates the complete FreshSky-Safe workflow:

- an unknown correlated channel law represented by a finite Bayesian ensemble;
- fixed multi-slot ACK/NACK delay and explicitly pending outcomes;
- latent receiver AoI and expected receiver-age propagation;
- an operator-selected per-device queue-cap admission shield;
- a deterministic excess-energy certificate for every time interval;
- powered dwell-delay, calibration, abrupt-shift, measured-replay, factorial,
  service-atlas, and runtime/memory experiments; and
- common-random-number pairing and seed-level confidence intervals.

Each manuscript-facing numerical claim is linked to committed seed-level data
and checked by `verify_claims.py`.

## Directory map

```text
iotj_open_source/
|-- README.md
|-- LICENSE
|-- THIRD_PARTY_NOTICES.md
|-- CITATION.cff
|-- requirements.txt
|-- reproduce.py             # portable quick/full entry point
|-- reproduce.ps1            # Windows wrapper
|-- reproduce.sh             # Linux/macOS wrapper
|-- verify_claims.py         # numerical and provenance gate
|-- analysis/                # IoTJ audits and figure generators
|-- experiments/             # simulator, data, and committed journal outputs
|-- iotj_evidence/           # prediction/action and regularization seed data
`-- figures/                 # committed vector and preview figures
```

Some internal files retain a `tmc_` prefix from the journal-development stage.
Those stable identifiers preserve script and data-schema compatibility;
`iotj_evidence/` and this README define the public release surface.

## Environment

The reported runs used CPython 3.12.7 on Windows 11. Create a fresh environment:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS, install the matching CPU build of PyTorch if the exact
`+cpu` wheel is not available from the default index. The quick validation path
does not train the optional deep-index baseline, but the full experiment suite
does.

## Quick reproduction

The quick path validates committed seed outputs, checks the measured-data hash,
recomputes all manuscript-facing claims, and rebuilds the empirical figures:

```bash
python reproduce.py --quick
```

Windows PowerShell:

```powershell
.\reproduce.ps1
```

A successful run ends with:

```text
PASS: all manuscript-facing numerical and provenance checks passed
FreshSky-Safe reproduction completed successfully
```

## Full reproduction

The full path reruns the journal simulation suites before checking claims:

```bash
python reproduce.py --full
```

or:

```powershell
.\reproduce.ps1 -Full
```

This path is CPU intensive because it includes the 20-seed policy grids,
factorial attribution, nine-point queue-cap atlas, measurement replay, runtime
grid, 540-run closed-loop audit, and held-out regularization study. Fixed seeds
are committed in the scripts.

## Paper-to-artifact map

| Manuscript evidence | Source or check | Primary committed output |
|---|---|---|
| Bayesian prediction calibration and law shift | `analysis/iotj_inference_evidence.py` | `iotj_evidence/stationary_prediction_*.csv`, `shift_prediction_*.csv` |
| Bayesian-to-action grid | `analysis/iotj_closed_loop_grid.py` | `iotj_evidence/closed_loop_grid_*.csv` |
| Held-out geometry anchoring | `analysis/iotj_regularized_policy_selection.py` | `iotj_evidence/regularized_*.csv` |
| Expected receiver-AoI grid | `experiments/tmc_calibration_strict.py` | `experiments/results/tmc_aoi_calibration_*.csv` |
| Recent baselines and factorial attribution | `experiments/tmc_extended_evidence.py` | `experiments/results/tmc_extended/` |
| Measured urban A2G replay | `experiments/tmc_measured_strict.py` | `experiments/results/measured_a2g_delayed_aoi_*.csv` |
| Queue-cap service atlas | `experiments/tmc_extended_evidence.py` | `experiments/results/tmc_extended/tmc_qcap_service_*.csv` |
| Runtime/memory envelope | `experiments/tmc_runtime_scaling.py` | `experiments/results/tmc_runtime_scaling.csv` |
| Headline numerical gate | `verify_claims.py` | `verification/claim_check_report.json` |

## Data and licenses

- Source code is released under the MIT License.
- `experiments/data/measured_a2g_urban_3p6ghz.xlsx` is a verified copy of the
  CC BY 4.0 dataset identified in `THIRD_PARTY_NOTICES.md`.
- Cached OpenStreetMap inputs remain subject to the Open Database License and
  include OpenStreetMap contributor attribution.
- Generated CSV results and original figures may be reused with attribution to
  the accompanying paper and this artifact.

The measured replay uses one time-ordered UAV-ground trace phase-shifted across
simulated devices. It is not a multi-UAV measurement campaign or a
hardware-in-the-loop experiment.

## Integrity

`MANIFEST.sha256` records every released file after packaging. Run
`python make_manifest.py --check` and `python verify_claims.py` before release
and after download. If any licensed third-party
file is removed for hosting-policy reasons, retain the notice and download it
from its DOI using the recorded SHA-256 before full reproduction.
