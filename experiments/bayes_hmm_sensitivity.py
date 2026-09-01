# -*- coding: utf-8 -*-
"""Forgetting-factor sensitivity for the safe Bayesian FreshSky learner.

The experiment reuses identical environments and seeds from the stationary
and abrupt-shift evaluations.  Only the Bayesian model-weight forgetting
factor changes; qcap, budget, model grid, and scheduling objective are fixed.
"""
import os
import pandas as pd

import belief_whittle as B
import bayes_hmm_experiment as E
import bayes_hmm_shift as S


FORGETTING = (0.0, 5e-4, 2e-3, 1e-2, 2e-2)
SEEDS = (0, 1, 2, 3, 4)


def policy_class(forget):
    """Create a safe policy class with one fixed forgetting factor."""

    class SafeBayesForget(E.BayesFreshSkySafe):
        name = f"Bayes-FreshSky-Safe-f{forget:g}"

        def __init__(self, cfg):
            super().__init__(cfg)
            self.filter = E.BayesianGEFilter(cfg.N, forget=forget)

    return SafeBayesForget


def stationary_rows(cfg):
    summaries = []
    seeds = []
    for forget in FORGETTING:
        frame, summary = E.run_detailed(
            policy_class(forget), cfg, seeds=SEEDS
        )
        frame.insert(0, "forget", forget)
        frame.insert(0, "scenario", "stationary-dwell8")
        seeds.append(frame)
        summaries.append(
            dict(
                scenario="stationary-dwell8",
                forget=forget,
                wAoI=summary["wAoI"],
                wAoI_std=summary["wAoI_std"],
                max_avg_power=summary["max_avg_power"],
                max_queue=summary["max_queue"],
                model_entropy=summary["mean_model_entropy"],
                feasible_all_seeds=summary["feasible_all_seeds"],
                certificate_all_seeds=summary["certificate_all_seeds"],
            )
        )
    return seeds, summaries


def shift_rows(cfg):
    summaries = []
    seeds = []
    for forget in FORGETTING:
        rows = [
            S.run_one(policy_class(forget), cfg, seed) for seed in SEEDS
        ]
        frame = pd.DataFrame(rows)
        frame.insert(0, "forget", forget)
        frame.insert(0, "scenario", "shift-dwell2-to14")
        seeds.append(frame)
        summaries.append(
            dict(
                scenario="shift-dwell2-to14",
                forget=forget,
                pre_wAoI=frame.pre_wAoI.mean(),
                post_wAoI=frame.post_wAoI.mean(),
                post_wAoI_std=frame.post_wAoI.std(ddof=1),
                max_avg_power=frame.post_max_avg_power.mean(),
                max_queue=frame.max_queue.max(),
                model_entropy=frame.final_model_entropy.mean(),
                feasible_all_seeds=frame.post_feasible.all(),
                certificate_all_seeds=frame.certificate_ok.all(),
            )
        )
    return seeds, summaries


def main():
    cfg = E.clone_cfg(T=4000, pbar=0.4, qcap=24.0, mean_dwell=8.0)
    stationary_seed, stationary_summary = stationary_rows(cfg)
    shift_seed, shift_summary = shift_rows(cfg)

    os.makedirs(B.RES, exist_ok=True)
    summary_path = os.path.join(
        B.RES, "bayes_hmm_sensitivity_summary.csv"
    )
    seed_path = os.path.join(B.RES, "bayes_hmm_sensitivity_seeds.csv")
    pd.DataFrame(stationary_summary + shift_summary).to_csv(
        summary_path, index=False
    )
    pd.concat(stationary_seed + shift_seed, ignore_index=True).to_csv(
        seed_path, index=False
    )

    table = pd.DataFrame(stationary_summary + shift_summary)
    print(table.to_string(index=False))
    print(f"saved {summary_path} and {seed_path}")


if __name__ == "__main__":
    main()
