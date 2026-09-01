# -*- coding: utf-8 -*-
"""Paired-seed analysis for measured delayed-AoI replay."""
import os
import numpy as np
import pandas as pd

import belief_whittle as B


INPUT = os.path.join(B.RES, "measured_a2g_delayed_aoi_seeds.csv")
OUTPUT = os.path.join(B.RES, "measured_a2g_delayed_aoi_paired.csv")
T_CRIT_95_DF4 = 2.7764451051977987


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    half = T_CRIT_95_DF4 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean, half


def main():
    frame = pd.read_csv(INPUT)
    pivot = frame.pivot(
        index=["quantile", "good_fraction", "extra_delay", "seed"],
        columns="age_mode",
        values="wAoI",
    ).reset_index()
    rows = []
    for keys, part in pivot.groupby(
        ["quantile", "good_fraction", "extra_delay"]
    ):
        gain = 100.0 * (
            part["conservative"] - part["expected"]
        ) / part["conservative"]
        true_gap = 100.0 * (
            part["expected"] - part["perfect"]
        ) / part["perfect"]
        gain_mean, gain_ci = mean_ci(gain)
        gap_mean, gap_ci = mean_ci(true_gap)
        rows.append(
            dict(
                quantile=keys[0],
                good_fraction=keys[1],
                extra_delay=keys[2],
                expected_gain_vs_conservative_pct=gain_mean,
                expected_gain_ci95_halfwidth=gain_ci,
                expected_gap_vs_true_aoi_pct=gap_mean,
                expected_gap_vs_true_aoi_ci95_halfwidth=gap_ci,
                paired_seeds=len(part),
            )
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
