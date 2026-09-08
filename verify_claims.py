#!/usr/bin/env python3
"""Verify the numerical claims and provenance of the IoTJ artifact."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "experiments" / "results"
IOTJ = ROOT / "iotj_evidence"
MEASURED = ROOT / "experiments" / "data" / "measured_a2g_urban_3p6ghz.xlsx"


def close(label: str, observed: float, expected: float, tolerance: float = 0.011) -> None:
    if not math.isclose(float(observed), expected, abs_tol=tolerance):
        raise AssertionError(f"{label}: observed {observed}, expected {expected}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checks: dict[str, object] = {}

    expected_hash = "19efd11676921eb115f16df4c18f78d54f13ce5a65876baefab37f204604c5d3"
    observed_hash = sha256(MEASURED)
    if observed_hash != expected_hash:
        raise AssertionError("measured CIR SHA-256 mismatch")
    checks["measured_cir_sha256"] = observed_hash

    stationary = pd.read_csv(IOTJ / "stationary_prediction_summary.csv")
    prediction = stationary[(stationary.dwell == 14) & (stationary.delay == 2)].iloc[0]
    close("Brier skill", prediction.brier_skill_vs_static_pct, 12.39)
    close("Brier skill CI", prediction.brier_skill_vs_static_pct_ci95, 1.08)
    checks["brier_skill_dwell14_delay2_pct"] = float(
        prediction.brier_skill_vs_static_pct
    )

    shift = pd.read_csv(IOTJ / "shift_prediction_summary.csv")
    late = shift[
        (shift.metric == "bayes_skill_vs_fixed_pct") & (shift.phase == "post_late")
    ].iloc[0]
    close("late-shift Brier skill", late["mean"], 2.37)
    close("late-shift Brier CI", late.ci95, 0.57)

    closed = pd.read_csv(IOTJ / "closed_loop_grid_seeds.csv")
    if len(closed) != 540 or not closed.certificate_ok.astype(bool).all():
        raise AssertionError("closed-loop certificates are not 540/540")
    checks["closed_loop_certificates"] = "540/540"

    age = pd.read_csv(RESULTS / "tmc_aoi_calibration_paired.csv")
    age_min = float(age.gain_vs_conservative_pct.min())
    age_max = float(age.gain_vs_conservative_pct.max())
    close("synthetic expected-age minimum", age_min, 14.28)
    close("synthetic expected-age maximum", age_max, 41.28)
    checks["synthetic_expected_age_gain_pct"] = [age_min, age_max]

    measured = pd.read_csv(RESULTS / "measured_a2g_delayed_aoi_paired.csv")
    measured_min = float(measured.expected_gain_vs_conservative_pct.min())
    measured_max = float(measured.expected_gain_vs_conservative_pct.max())
    close("measured expected-age minimum", measured_min, 8.04)
    close("measured expected-age maximum", measured_max, 36.89)
    measured_seeds = pd.read_csv(RESULTS / "measured_a2g_delayed_aoi_seeds.csv")
    if len(measured_seeds) != 360 or not measured_seeds.certificate_ok.astype(bool).all():
        raise AssertionError("measured-replay certificates are not 360/360")
    checks["measured_expected_age_gain_pct"] = [measured_min, measured_max]
    checks["measured_replay_certificates"] = "360/360"

    qcap = pd.read_csv(RESULTS / "tmc_extended" / "tmc_qcap_service_seeds.csv")
    if len(qcap) != 180 or qcap.qcap.nunique() != 9 or not qcap.certificate_ok.astype(bool).all():
        raise AssertionError("queue-cap atlas is not nine points with 180/180 certificates")
    checks["queue_cap_atlas"] = {"settings": 9, "certificates": "180/180"}

    runtime = pd.read_csv(RESULTS / "tmc_runtime_scaling.csv")
    point = runtime[
        (runtime.N == 500)
        & (runtime.retained_models == 144)
        & (runtime.extra_delay == 8)
    ].iloc[0]
    close("runtime median ms", point.total_median_us / 1000.0, 3.467)
    close("state memory MiB", point.total_memory_kib / 1024.0, 1.15)
    checks["runtime_N500_K144_d8"] = {
        "median_ms": float(point.total_median_us / 1000.0),
        "state_mib": float(point.total_memory_kib / 1024.0),
    }

    report_dir = ROOT / "verification"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "claim_check_report.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(checks, indent=2, sort_keys=True))
    print("PASS: all manuscript-facing numerical and provenance checks passed")


if __name__ == "__main__":
    main()
