# -*- coding: utf-8 -*-
"""Matched delayed-feedback adaptations of Ji'24 and Zhu'26.

These policies reproduce the closest public algorithmic ideas under the
FreshSky simulator's multi-UAV action, channel, and energy model.  They are
not exact reproductions of the papers' different CPOMDP/downlink systems.
"""
import os
import numpy as np
import pandas as pd

import belief_whittle as B
import bayes_hmm_experiment as E
import bayes_hmm_delayed_aoi_belief as A


DELAYS = (1, 2, 4, 8)
SEEDS = A.SEEDS


class Ji24ExpectedAoIGreedySafe(A.ExpectedAoIBayesSafe):
    """Expected immediate AoI reduction with the same Qmax safety shield."""

    name = "Ji'24-style Expected-AoI Greedy-Safe"

    def __call__(self, age, unused, queue, value, pth, cfg):
        del unused
        success_prob = self.success_probability()
        score = value * age * success_prob
        next_q = np.maximum(queue - cfg.pbar, 0.0) + pth
        score = np.where(next_q <= self.qcap + 1e-12, score, -1e18)
        return B.topM(score, cfg)


class Zhu26ExpectedAoIAggregateDPP(A.ExpectedAoIBayesSafe):
    """Expected-AoI DPP with one aggregate resource queue and no shield."""

    name = "Zhu'26-style Expected-AoI Aggregate-DPP"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.Z = 0.0

    def __call__(self, age, unused, queue, value, pth, cfg):
        del unused, queue
        success_prob = self.success_probability()
        score = cfg.V * value * age * success_prob - self.Z * pth
        return B.topM(score, cfg)

    def update(self, energy):
        self.Z = max(
            0.0,
            self.Z + float(np.sum(energy)) - self.cfg.N * self.cfg.pbar,
        )


POLICIES = (
    A.ExpectedAoIBayesSafe,
    Ji24ExpectedAoIGreedySafe,
    Zhu26ExpectedAoIAggregateDPP,
)


def main():
    cfg = E.clone_cfg(T=4000, pbar=0.4, qcap=24.0, mean_dwell=8.0)
    rows = []
    for policy in POLICIES:
        for delay in DELAYS:
            rows.extend(
                A.run_one(policy, cfg, delay, seed) for seed in SEEDS
            )
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
            finite_window_bound=("finite_window_bound", "first"),
            bound_all_seeds=("bound_ok", "all"),
            certificate_all_seeds=("certificate_ok", "all"),
        )
    )
    os.makedirs(B.RES, exist_ok=True)
    summary_path = os.path.join(
        B.RES, "delayed_aoi_latest_baselines_summary.csv"
    )
    seed_path = os.path.join(
        B.RES, "delayed_aoi_latest_baselines_seeds.csv"
    )
    summary.to_csv(summary_path, index=False)
    seeds.to_csv(seed_path, index=False)
    print(summary.to_string(index=False))
    print(f"saved {summary_path} and {seed_path}")


if __name__ == "__main__":
    main()
