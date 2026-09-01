#!/usr/bin/env python3
"""Closed-loop prediction-to-action test for the IoTJ-targeted manuscript.

The test uses the manuscript simulator and a pre-specified 3 x 3 grid of
channel dwell times and feedback delays.  Bayesian, static-geometry, and
known-transition schedulers receive matched random streams within each seed.
Positive paired skill means that Bayesian scheduling lowers W-AoI.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS = PROJECT / "iotj_evidence"
DWELLS = (2.0, 8.0, 14.0)
DELAYS = (2, 4, 8)
SEEDS = tuple(range(20))
POLICIES = ("Bayesian ensemble", "Static geometry", "Known transition")


def run_task(task: tuple[str, float, int, int]) -> dict:
    """Run one matched cell; imports occur once per spawned worker."""
    import sys

    experiments = PROJECT / "experiments"
    if str(experiments) not in sys.path:
        sys.path.insert(0, str(experiments))
    import tmc_extended_evidence as X

    policy_name, dwell, delay, seed = task
    B, E, D, A = X.load_project(PROJECT)
    ours, baselines, _ = X.build_policy_classes(B, E, D)
    policy_by_name = {
        "Bayesian ensemble": ours,
        "Static geometry": next(
            policy for policy in baselines if policy.name == "Static-Geometry Safe"
        ),
        "Known transition": next(
            policy for policy in baselines if policy.name == "Known-Transition Safe"
        ),
    }
    cfg = E.clone_cfg(T=4000, pbar=0.4, qcap=24.0, mean_dwell=dwell)
    row = X.run_one(B, E, A, policy_by_name[policy_name], cfg, delay, seed)
    row.update(method=policy_name, dwell=dwell, delay=delay)
    return row


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, 0.0
    # Two-sided 95% Student-t critical value for 19 degrees of freedom.
    ci = 2.093024054 * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return mean, ci


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    paired_rows = []
    for (dwell, delay, method), part in frame.groupby(
        ["dwell", "delay", "method"], sort=True
    ):
        w_mean, w_ci = mean_ci(part.wAoI.to_numpy())
        worst_mean, worst_ci = mean_ci(part.worst_user_mean_aoi.to_numpy())
        rows.append(
            {
                "dwell": dwell,
                "delay": delay,
                "method": method,
                "seeds": part.seed.nunique(),
                "wAoI": w_mean,
                "wAoI_ci95": w_ci,
                "worst_user_mean_aoi": worst_mean,
                "worst_user_mean_aoi_ci95": worst_ci,
                "certificate_all_seeds": bool(part.certificate_ok.all()),
            }
        )

    wide = frame.pivot_table(
        index=["dwell", "delay", "seed"], columns="method", values="wAoI"
    ).reset_index()
    for (dwell, delay), part in wide.groupby(["dwell", "delay"], sort=True):
        bayes_skill = 100.0 * (
            part["Static geometry"] - part["Bayesian ensemble"]
        ) / part["Static geometry"]
        bayes_gap_known = 100.0 * (
            part["Bayesian ensemble"] - part["Known transition"]
        ) / part["Known transition"]
        skill_mean, skill_ci = mean_ci(bayes_skill.to_numpy())
        gap_mean, gap_ci = mean_ci(bayes_gap_known.to_numpy())
        paired_rows.append(
            {
                "dwell": dwell,
                "delay": delay,
                "seeds": len(part),
                "bayesian_wAoI_skill_vs_static_pct": skill_mean,
                "bayesian_wAoI_skill_vs_static_pct_ci95": skill_ci,
                "bayesian_wAoI_gap_to_known_pct": gap_mean,
                "bayesian_wAoI_gap_to_known_pct_ci95": gap_ci,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(paired_rows)


def write_summary(paired: pd.DataFrame) -> None:
    lines = [
        "# Closed-loop cognition-to-action grid",
        "",
        "The grid, methods, simulator horizon, and 20-seed protocol were fixed before execution. Positive skill means that Bayesian scheduling lowers W-AoI relative to static geometry. Each comparison is paired by seed.",
        "",
        "| NLoS dwell | delay | Bayesian W-AoI skill vs static geometry | Bayesian W-AoI gap to known transition |",
        "|---:|---:|---:|---:|",
    ]
    for row in paired.itertuples(index=False):
        lines.append(
            f"| {row.dwell:.0f} | {row.delay:.0f} | "
            f"{row.bayesian_wAoI_skill_vs_static_pct:.2f}% ± "
            f"{row.bayesian_wAoI_skill_vs_static_pct_ci95:.2f}% | "
            f"{row.bayesian_wAoI_gap_to_known_pct:.2f}% ± "
            f"{row.bayesian_wAoI_gap_to_known_pct_ci95:.2f}% |"
        )
    lines.extend(
        [
            "",
            "The complete grid is reported to expose, rather than hide, regimes in which delayed Bayesian learning does not improve the closed-loop objective.",
            "",
        ]
    )
    (RESULTS / "closed_loop_grid_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    tasks = [
        (method, dwell, delay, seed)
        for method in POLICIES
        for dwell in DWELLS
        for delay in DELAYS
        for seed in SEEDS
    ]
    workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    print(f"Running {len(tasks)} matched simulations with {workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run_task, tasks, chunksize=2))
    seeds = pd.DataFrame(rows)
    aggregate, paired = summarize(seeds)
    seeds.to_csv(RESULTS / "closed_loop_grid_seeds.csv", index=False)
    aggregate.to_csv(RESULTS / "closed_loop_grid_aggregate.csv", index=False)
    paired.to_csv(RESULTS / "closed_loop_grid_paired.csv", index=False)
    write_summary(paired)
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
