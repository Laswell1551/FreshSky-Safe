"""V and forgetting-factor sensitivity with delayed expected-AoI control."""
from __future__ import annotations

import os
import pandas as pd

import bayes_hmm_delayed_aoi_belief as A
import bayes_hmm_experiment as E
import belief_whittle as B


V_VALUES = (2.0, 4.0, 6.0, 8.0, 12.0)
F_VALUES = (0.0, 5e-4, 2e-3, 1e-2, 2e-2)
SEEDS = (0, 1, 2, 3, 4)
DELAY = 4


def policy_with_forgetting(value: float):
    class ExpectedAoIWithForgetting(A.ExpectedAoIBayesSafe):
        name = f"Expected-AoI Bayes-Safe-f{value:g}"

        def __init__(self, cfg):
            super().__init__(cfg)
            self.filter = E.BayesianGEFilter(cfg.N, forget=value)

    return ExpectedAoIWithForgetting


def summarize(frame: pd.DataFrame, parameter: str) -> pd.DataFrame:
    return (
        frame.groupby(parameter, as_index=False)
        .agg(
            wAoI=("wAoI", "mean"),
            wAoI_std=("wAoI", "std"),
            max_avg_power=("max_avg_power", "mean"),
            max_queue=("max_queue", "max"),
            certificate_all_seeds=("certificate_ok", "all"),
        )
    )


def main() -> None:
    rows = []
    for value in V_VALUES:
        cfg = E.clone_cfg(
            T=4000,
            pbar=0.4,
            qcap=24.0,
            mean_dwell=8.0,
            V=value,
        )
        for seed in SEEDS:
            row = A.run_one(A.ExpectedAoIBayesSafe, cfg, DELAY, seed)
            row.update(sweep="V", value=value)
            rows.append(row)
    for value in F_VALUES:
        cfg = E.clone_cfg(
            T=4000,
            pbar=0.4,
            qcap=24.0,
            mean_dwell=8.0,
            V=6.0,
        )
        policy = policy_with_forgetting(value)
        for seed in SEEDS:
            row = A.run_one(policy, cfg, DELAY, seed)
            row.update(sweep="forget", value=value)
            rows.append(row)

    seeds = pd.DataFrame(rows)
    summaries = []
    for sweep, frame in seeds.groupby("sweep"):
        part = summarize(frame.rename(columns={"value": sweep}), sweep)
        part.insert(0, "sweep", sweep)
        part = part.rename(columns={sweep: "value"})
        summaries.append(part)
    summary = pd.concat(summaries, ignore_index=True)
    os.makedirs(B.RES, exist_ok=True)
    seeds.to_csv(os.path.join(B.RES, "tmc_parameter_sensitivity_seeds.csv"), index=False)
    summary.to_csv(os.path.join(B.RES, "tmc_parameter_sensitivity_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
