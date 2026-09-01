# -*- coding: utf-8 -*-
"""Higher-power validation for FreshSky-Safe.

Runs 20 matched seeds for the closest delayed-feedback adaptations and a
dwell-by-delay calibration of the expected receiver-AoI approximation.
External policies are matched adaptations, not exact reproductions.
"""
import os
import numpy as np
import pandas as pd

import belief_whittle as B
import bayes_hmm_experiment as E
import bayes_hmm_delayed_aoi_belief as A
import delayed_aoi_latest_baselines as L


HEADLINE_SEEDS = tuple(range(20))
CALIBRATION_SEEDS = tuple(range(10))
DELAYS = (2, 4, 8)
DWELLS = (2.0, 4.0, 8.0, 14.0)
T95_DF19 = 2.093024054408263
T95_DF9 = 2.2621571627409915

OURS = "Expected-AoI Bayes-Safe"
CONSERVATIVE = "Conservative-AoI Bayes-Safe"
PERFECT = "Perfect-AoI Bayes-Safe"
JI = "Ji'24-style Expected-AoI Greedy-Safe"
ZHU = "Zhu'26-style Expected-AoI Aggregate-DPP"


def mean_ci(values, critical):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    half = float(critical * np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, half


def headline_validation():
    cfg = E.clone_cfg(T=4000, pbar=0.4, qcap=24.0, mean_dwell=8.0)
    policies = (
        A.ExpectedAoIBayesSafe,
        L.Ji24ExpectedAoIGreedySafe,
        L.Zhu26ExpectedAoIAggregateDPP,
    )
    rows = [
        A.run_one(policy, cfg, delay, seed)
        for policy in policies
        for delay in DELAYS
        for seed in HEADLINE_SEEDS
    ]
    seeds = pd.DataFrame(rows)
    summary = (
        seeds.groupby(["policy", "extra_delay"], as_index=False)
        .agg(
            wAoI=("wAoI", "mean"),
            wAoI_std=("wAoI", "std"),
            age_mae=("age_mae", "mean"),
            age_bias=("age_bias", "mean"),
            max_avg_power=("max_avg_power", "mean"),
            max_queue=("max_queue", "max"),
            bound_all_seeds=("bound_ok", "all"),
            certificate_all_seeds=("certificate_ok", "all"),
        )
    )
    age = seeds.pivot(
        index=["extra_delay", "seed"], columns="policy", values="wAoI"
    )
    power = seeds.pivot(
        index=["extra_delay", "seed"],
        columns="policy",
        values="max_avg_power",
    )
    paired_rows = []
    for delay in DELAYS:
        a = age.loc[delay]
        p = power.loc[delay]
        gain_ji = 100.0 * (a[JI] - a[OURS]) / a[JI]
        penalty_zhu = 100.0 * (a[OURS] - a[ZHU]) / a[ZHU]
        reduction_zhu = 100.0 * (p[ZHU] - p[OURS]) / p[ZHU]
        gain_mean, gain_ci = mean_ci(gain_ji, T95_DF19)
        penalty_mean, penalty_ci = mean_ci(penalty_zhu, T95_DF19)
        reduction_mean, reduction_ci = mean_ci(reduction_zhu, T95_DF19)
        paired_rows.append(
            dict(
                extra_delay=delay,
                gain_vs_ji24_style_pct=gain_mean,
                gain_vs_ji24_ci95_halfwidth=gain_ci,
                aoi_penalty_vs_zhu26_style_pct=penalty_mean,
                aoi_penalty_vs_zhu26_ci95_halfwidth=penalty_ci,
                max_power_reduction_vs_zhu26_pct=reduction_mean,
                max_power_reduction_vs_zhu26_ci95_halfwidth=reduction_ci,
                paired_seeds=len(a),
            )
        )
    return seeds, summary, pd.DataFrame(paired_rows)


def calibration_validation():
    policies = (
        A.ConservativeBayesSafe,
        A.ExpectedAoIBayesSafe,
        A.PerfectAoIBayesSafe,
    )
    rows = []
    for dwell in DWELLS:
        cfg = E.clone_cfg(
            T=4000, pbar=0.4, qcap=24.0, mean_dwell=dwell
        )
        for policy in policies:
            for delay in DELAYS:
                for seed in CALIBRATION_SEEDS:
                    row = A.run_one(policy, cfg, delay, seed)
                    row["mean_dwell"] = dwell
                    rows.append(row)
    seeds = pd.DataFrame(rows)
    summary = (
        seeds.groupby(
            ["mean_dwell", "extra_delay", "policy", "age_mode"],
            as_index=False,
        )
        .agg(
            wAoI=("wAoI", "mean"),
            wAoI_std=("wAoI", "std"),
            age_mae=("age_mae", "mean"),
            age_bias=("age_bias", "mean"),
            max_avg_power=("max_avg_power", "mean"),
            max_queue=("max_queue", "max"),
            bound_all_seeds=("bound_ok", "all"),
            certificate_all_seeds=("certificate_ok", "all"),
        )
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
            gap = 100.0 * (
                part[OURS] - part[PERFECT]
            ) / part[PERFECT]
            gain_mean, gain_ci = mean_ci(gain, T95_DF9)
            gap_mean, gap_ci = mean_ci(gap, T95_DF9)
            expected_rows = seeds[
                (seeds.mean_dwell == dwell)
                & (seeds.extra_delay == delay)
                & (seeds.policy == OURS)
            ]
            paired_rows.append(
                dict(
                    mean_dwell=dwell,
                    extra_delay=delay,
                    gain_vs_conservative_pct=gain_mean,
                    gain_vs_conservative_ci95_halfwidth=gain_ci,
                    gap_vs_true_aoi_pct=gap_mean,
                    gap_vs_true_aoi_ci95_halfwidth=gap_ci,
                    expected_age_mae=expected_rows.age_mae.mean(),
                    expected_age_bias=expected_rows.age_bias.mean(),
                    certificate_all_seeds=expected_rows.certificate_ok.all(),
                    paired_seeds=len(part),
                )
            )
    return seeds, summary, pd.DataFrame(paired_rows)


def main():
    os.makedirs(B.RES, exist_ok=True)
    h_seed, h_summary, h_paired = headline_validation()
    c_seed, c_summary, c_paired = calibration_validation()
    outputs = {
        "tmc_headline_20seed_seeds.csv": h_seed,
        "tmc_headline_20seed_summary.csv": h_summary,
        "tmc_headline_20seed_paired.csv": h_paired,
        "tmc_aoi_calibration_seeds.csv": c_seed,
        "tmc_aoi_calibration_summary.csv": c_summary,
        "tmc_aoi_calibration_paired.csv": c_paired,
    }
    for name, frame in outputs.items():
        frame.to_csv(os.path.join(B.RES, name), index=False)
    print("\n=== 20-seed headline comparison ===")
    print(h_summary.to_string(index=False))
    print("\n=== 20-seed paired effects ===")
    print(h_paired.to_string(index=False))
    print("\n=== dwell-delay AoI calibration ===")
    print(c_paired.to_string(index=False))
    print("\nSaved:")
    for name in outputs:
        print(os.path.join(B.RES, name))


if __name__ == "__main__":
    main()
