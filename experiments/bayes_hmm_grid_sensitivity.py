# -*- coding: utf-8 -*-
"""Bayesian transition-grid sensitivity under stationary and shifted channels."""
import os
import numpy as np
import pandas as pd

import belief_whittle as B
import bayes_hmm_experiment as E
import bayes_hmm_shift as S


GRID_SIZES = (6, 12, 20)
SEEDS = (0, 1, 2, 3, 4)
FORGET = 2e-3


class GridFilter(E.BayesianGEFilter):
    def __init__(self, n_uav, grid_size):
        pll = np.linspace(0.55, 0.99, grid_size)
        pnl = np.geomspace(0.005, 0.45, grid_size)
        self.pLL, self.pNL = np.meshgrid(pll, pnl, indexing="ij")
        self.pLL = self.pLL.ravel()
        self.pNL = self.pNL.ravel()
        keep = self.pLL > self.pNL
        self.pLL, self.pNL = self.pLL[keep], self.pNL[keep]
        self.K = len(self.pLL)
        self.N = n_uav
        self.forget = FORGET
        self.weight = np.full((self.N, self.K), 1.0 / self.K)
        self.theta = np.full((self.N, self.K), 0.5)


def policy_class(grid_size):
    class SafeBayesGrid(E.BayesFreshSkySafe):
        name = f"Bayes-FreshSky-Safe-grid{grid_size}"

        def __init__(self, cfg):
            super().__init__(cfg)
            self.filter = GridFilter(cfg.N, grid_size)

    return SafeBayesGrid


def main():
    cfg = E.clone_cfg(T=4000, pbar=0.4, qcap=24.0, mean_dwell=8.0)
    summary_rows = []
    seed_frames = []
    for grid_size in GRID_SIZES:
        policy = policy_class(grid_size)
        frame, summary = E.run_detailed(policy, cfg, seeds=SEEDS)
        frame.insert(0, "grid_size", grid_size)
        frame.insert(0, "scenario", "stationary-dwell8")
        seed_frames.append(frame)
        summary_rows.append(
            dict(
                scenario="stationary-dwell8",
                grid_size=grid_size,
                models=GridFilter(cfg.N, grid_size).K,
                wAoI=summary["wAoI"],
                wAoI_std=summary["wAoI_std"],
                max_avg_power=summary["max_avg_power"],
                max_queue=summary["max_queue"],
                feasible_all_seeds=summary["feasible_all_seeds"],
                certificate_all_seeds=summary["certificate_all_seeds"],
            )
        )

        shifted = pd.DataFrame(
            [S.run_one(policy, cfg, seed) for seed in SEEDS]
        )
        shifted.insert(0, "grid_size", grid_size)
        shifted.insert(0, "scenario", "shift-dwell2-to14")
        seed_frames.append(shifted)
        summary_rows.append(
            dict(
                scenario="shift-dwell2-to14",
                grid_size=grid_size,
                models=GridFilter(cfg.N, grid_size).K,
                pre_wAoI=shifted.pre_wAoI.mean(),
                post_wAoI=shifted.post_wAoI.mean(),
                post_wAoI_std=shifted.post_wAoI.std(ddof=1),
                max_avg_power=shifted.post_max_avg_power.mean(),
                max_queue=shifted.max_queue.max(),
                feasible_all_seeds=shifted.post_feasible.all(),
                certificate_all_seeds=shifted.certificate_ok.all(),
            )
        )

    os.makedirs(B.RES, exist_ok=True)
    summary_path = os.path.join(
        B.RES, "bayes_hmm_grid_sensitivity_summary.csv"
    )
    seed_path = os.path.join(
        B.RES, "bayes_hmm_grid_sensitivity_seeds.csv"
    )
    summaries = pd.DataFrame(summary_rows)
    summaries.to_csv(summary_path, index=False)
    pd.concat(seed_frames, ignore_index=True).to_csv(seed_path, index=False)
    print(summaries.to_string(index=False))
    print(f"saved {summary_path} and {seed_path}")


if __name__ == "__main__":
    main()
