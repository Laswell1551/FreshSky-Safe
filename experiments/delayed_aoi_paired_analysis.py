# -*- coding: utf-8 -*-
"""Paired-seed analysis for delayed receiver-AoI estimators."""
import os
import numpy as np
import pandas as pd

import belief_whittle as B


INPUT = os.path.join(B.RES, "bayes_hmm_delayed_aoi_belief_seeds.csv")
OUTPUT = os.path.join(B.RES, "bayes_hmm_delayed_aoi_paired.csv")
T_CRIT_95_DF4 = 2.7764451051977987


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, np.nan
    half = T_CRIT_95_DF4 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean, half


def main():
    frame = pd.read_csv(INPUT)
    bayes_policies = {
        "Conservative-AoI Bayes-Safe",
        "Expected-AoI Bayes-Safe",
        "Perfect-AoI Bayes-Safe",
    }
    frame = frame[frame.policy.isin(bayes_policies)].copy()
    pivot = frame.pivot(
        index=["extra_delay", "seed"], columns="age_mode", values="wAoI"
    ).reset_index()
    baseline0 = (
        frame[
            (frame.age_mode == "expected") & (frame.extra_delay == 0)
        ]
        .set_index("seed")
        .wAoI
    )
    rows = []
    for delay in sorted(pivot.extra_delay.unique()):
        part = pivot[pivot.extra_delay == delay].set_index("seed")
        gain = 100.0 * (
            part["conservative"] - part["expected"]
        ) / part["conservative"]
        true_gap = 100.0 * (
            part["expected"] - part["perfect"]
        ) / part["perfect"]
        degradation = 100.0 * (
            part["expected"] - baseline0.loc[part.index]
        ) / baseline0.loc[part.index]
        gain_mean, gain_ci = mean_ci(gain)
        true_mean, true_ci = mean_ci(true_gap)
        deg_mean, deg_ci = mean_ci(degradation)
        rows.append(
            dict(
                extra_delay=delay,
                expected_gain_vs_conservative_pct=gain_mean,
                expected_gain_ci95_halfwidth=gain_ci,
                expected_gap_vs_true_aoi_pct=true_mean,
                expected_gap_vs_true_aoi_ci95_halfwidth=true_ci,
                expected_degradation_vs_delay0_pct=deg_mean,
                expected_degradation_ci95_halfwidth=deg_ci,
                paired_seeds=len(part),
            )
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
