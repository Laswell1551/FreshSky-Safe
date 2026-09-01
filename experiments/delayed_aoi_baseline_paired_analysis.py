# -*- coding: utf-8 -*-
"""Paired comparison against Ji'24- and Zhu'26-style adaptations."""
import os
import numpy as np
import pandas as pd

import belief_whittle as B


INPUT = os.path.join(B.RES, "delayed_aoi_latest_baselines_seeds.csv")
OUTPUT = os.path.join(B.RES, "delayed_aoi_latest_baselines_paired.csv")
T_CRIT_95_DF4 = 2.7764451051977987

OURS = "Expected-AoI Bayes-Safe"
JI = "Ji'24-style Expected-AoI Greedy-Safe"
ZHU = "Zhu'26-style Expected-AoI Aggregate-DPP"


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    half = T_CRIT_95_DF4 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean, half


def main():
    frame = pd.read_csv(INPUT)
    age = frame.pivot(
        index=["extra_delay", "seed"], columns="policy", values="wAoI"
    ).reset_index()
    power = frame.pivot(
        index=["extra_delay", "seed"],
        columns="policy",
        values="max_avg_power",
    ).reset_index()
    rows = []
    for delay in sorted(age.extra_delay.unique()):
        a = age[age.extra_delay == delay].set_index("seed")
        p = power[power.extra_delay == delay].set_index("seed")
        gain_ji = 100.0 * (a[JI] - a[OURS]) / a[JI]
        penalty_zhu = 100.0 * (a[OURS] - a[ZHU]) / a[ZHU]
        power_reduction_zhu = 100.0 * (p[ZHU] - p[OURS]) / p[ZHU]
        gain_mean, gain_ci = mean_ci(gain_ji)
        penalty_mean, penalty_ci = mean_ci(penalty_zhu)
        power_mean, power_ci = mean_ci(power_reduction_zhu)
        rows.append(
            dict(
                extra_delay=delay,
                gain_vs_ji24_style_pct=gain_mean,
                gain_vs_ji24_ci95_halfwidth=gain_ci,
                aoi_penalty_vs_zhu26_style_pct=penalty_mean,
                aoi_penalty_vs_zhu26_ci95_halfwidth=penalty_ci,
                max_power_reduction_vs_zhu26_pct=power_mean,
                max_power_reduction_vs_zhu26_ci95_halfwidth=power_ci,
                paired_seeds=len(a),
            )
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
