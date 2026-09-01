# -*- coding: utf-8 -*-
"""Strict common-random-number correlation-delay calibration.

This companion uses the corrected stream-separated runner in
tmc_extended_evidence.py and exports 10-seed expected/conservative/perfect
receiver-age comparisons over the 4x3 dwell-delay grid.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import tmc_extended_evidence as X


DWELLS = (2.0, 4.0, 8.0, 14.0)
DELAYS = (2, 4, 8)
SEEDS = tuple(range(10))
OURS = "Expected-AoI Bayes-Safe"
CONSERVATIVE = "Conservative-AoI Bayes-Safe"
PERFECT = "Perfect-AoI Bayes-Safe"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    B, E, D, A = X.load_project(args.project)
    OursClass, baseline_policies, _ = X.build_policy_classes(B, E, D)
    ConservativeClass = next(
        policy
        for policy in baseline_policies
        if policy.name == "Conservative-AoI Safe"
    )

    class ExpectedClass(OursClass):
        name = OURS

    class ConservativeClassNamed(ConservativeClass):
        name = CONSERVATIVE

    class PerfectClass(OursClass):
        name = PERFECT
        age_mode = "perfect"

    policies = (
        ConservativeClassNamed,
        ExpectedClass,
        PerfectClass,
    )
    rows = []
    for dwell in DWELLS:
        cfg = E.clone_cfg(
            T=4000, pbar=0.4, qcap=24.0, mean_dwell=dwell
        )
        for policy in policies:
            for delay in DELAYS:
                for seed in SEEDS:
                    row = X.run_one(B, E, A, policy, cfg, delay, seed)
                    row["mean_dwell"] = dwell
                    rows.append(row)
    seeds = pd.DataFrame(rows)
    summary = X.summarize(
        seeds, ["mean_dwell", "extra_delay", "policy"]
    )

    pivot = seeds.pivot(
        index=["mean_dwell", "extra_delay", "seed"],
        columns="policy",
        values="wAoI",
    )
    paired_rows = []
    for dwell in DWELLS:
        for delay in DELAYS:
            part = pivot.loc[(dwell, delay)]
            gain = 100.0 * (
                part[CONSERVATIVE] - part[OURS]
            ) / part[CONSERVATIVE]
            gap = 100.0 * (part[OURS] - part[PERFECT]) / part[PERFECT]
            gain_mean, gain_ci = X.mean_ci95(gain)
            gap_mean, gap_ci = X.mean_ci95(gap)
            expected_rows = seeds[
                (seeds["mean_dwell"] == dwell)
                & (seeds["extra_delay"] == delay)
                & (seeds["policy"] == OURS)
            ]
            paired_rows.append(
                {
                    "mean_dwell": dwell,
                    "extra_delay": delay,
                    "gain_vs_conservative_pct": gain_mean,
                    "gain_vs_conservative_ci95_halfwidth": gain_ci,
                    "gap_vs_true_aoi_pct": gap_mean,
                    "gap_vs_true_aoi_ci95_halfwidth": gap_ci,
                    "expected_p99_slot_aoi": expected_rows[
                        "p99_slot_aoi"
                    ].mean(),
                    "expected_cvar95_slot_aoi": expected_rows[
                        "cvar95_slot_aoi"
                    ].mean(),
                    "certificate_all_seeds": expected_rows[
                        "certificate_ok"
                    ].all(),
                    "paired_seeds": len(part),
                }
            )
    paired = pd.DataFrame(paired_rows)

    outputs = {
        "tmc_aoi_calibration_seeds.csv": seeds,
        "tmc_aoi_calibration_summary.csv": summary,
        "tmc_aoi_calibration_paired.csv": paired,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output / name, index=False)
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
