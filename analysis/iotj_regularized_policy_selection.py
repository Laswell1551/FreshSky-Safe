#!/usr/bin/env python3
"""Validation/test study of geometry-anchored Bayesian decision beliefs.

A single global shrinkage coefficient alpha forms

    b_dec = (1-alpha) b_geometry + alpha b_Bayes.

The coefficient is selected on seeds 0--9 over the full 3 x 3 dwell-delay
grid, then evaluated once on untouched seeds 20--39.  Alpha=0 is included so
the procedure can reject online belief use rather than force a positive claim.
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
ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
VALIDATION_SEEDS = tuple(range(10))
TEST_SEEDS = tuple(range(20, 40))


def run_task(task: tuple[str, float, float, int, int]) -> dict:
    import sys

    stage, alpha, dwell, delay, seed = task
    experiments = PROJECT / "experiments"
    if str(experiments) not in sys.path:
        sys.path.insert(0, str(experiments))
    import tmc_extended_evidence as X

    B, E, D, A = X.load_project(PROJECT)
    ours, _, _ = X.build_policy_classes(B, E, D)
    policy = type(
        f"AnchoredBayes{str(alpha).replace('.', 'p')}",
        (ours,),
        {
            "name": f"Anchored Bayes (alpha={alpha:g})",
            "belief_weight": float(alpha),
        },
    )
    cfg = E.clone_cfg(T=4000, pbar=0.4, qcap=24.0, mean_dwell=dwell)
    row = X.run_one(B, E, A, policy, cfg, delay, seed)
    row.update(stage=stage, alpha=alpha, dwell=dwell, delay=delay)
    return row


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    critical_by_n = {10: 2.262157163, 20: 2.093024054}
    if len(values) not in critical_by_n:
        raise ValueError(f"No pre-specified t critical value for n={len(values)}")
    critical = critical_by_n[len(values)]
    ci = critical * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return mean, ci


def select_alpha(validation: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    wide = validation.pivot_table(
        index=["dwell", "delay", "seed"], columns="alpha", values="wAoI"
    )
    rows = []
    for alpha in ALPHAS:
        paired = 100.0 * (wide[0.0] - wide[alpha]) / wide[0.0]
        # A seed generates all nine dwell-delay cells, so the seed—not the
        # cell-seed row—is the independent unit for the across-grid interval.
        seed_level = paired.groupby(level="seed").mean()
        mean, ci = mean_ci(seed_level.to_numpy())
        rows.append(
            {
                "alpha": alpha,
                "mean_wAoI_skill_vs_alpha0_pct": mean,
                "ci95": ci,
                "validation_seeds": len(seed_level),
            }
        )
    summary = pd.DataFrame(rows)
    # Alpha=0 remains eligible: the validation procedure may reject learning.
    chosen = float(
        summary.sort_values(
            ["mean_wAoI_skill_vs_alpha0_pct", "alpha"],
            ascending=[False, True],
        ).iloc[0].alpha
    )
    return chosen, summary


def summarize_test(test: pd.DataFrame, chosen: float) -> pd.DataFrame:
    wide = test.pivot_table(
        index=["dwell", "delay", "seed"], columns="alpha", values="wAoI"
    ).reset_index()
    rows = []
    for (dwell, delay), part in wide.groupby(["dwell", "delay"], sort=True):
        skill = 100.0 * (part[0.0] - part[chosen]) / part[0.0]
        gain_raw = 100.0 * (part[1.0] - part[chosen]) / part[1.0]
        skill_mean, skill_ci = mean_ci(skill.to_numpy())
        raw_mean, raw_ci = mean_ci(gain_raw.to_numpy())
        rows.append(
            {
                "dwell": dwell,
                "delay": delay,
                "chosen_alpha": chosen,
                "seeds": len(part),
                "chosen_skill_vs_static_pct": skill_mean,
                "chosen_skill_vs_static_pct_ci95": skill_ci,
                "chosen_gain_vs_unregularized_pct": raw_mean,
                "chosen_gain_vs_unregularized_pct_ci95": raw_ci,
            }
        )
    return pd.DataFrame(rows)


def summarize_overall_test(test: pd.DataFrame, chosen: float) -> pd.DataFrame:
    """Across-grid test summary with seeds as the independent units."""
    wide = test.pivot_table(
        index=["dwell", "delay", "seed"], columns="alpha", values="wAoI"
    )
    per_cell_seed = pd.DataFrame(
        {
            "skill_vs_static_pct": 100.0 * (wide[0.0] - wide[chosen]) / wide[0.0],
            "gain_vs_unregularized_pct": 100.0
            * (wide[1.0] - wide[chosen])
            / wide[1.0],
        }
    )
    seed_level = per_cell_seed.groupby(level="seed").mean()
    static_mean, static_ci = mean_ci(seed_level.skill_vs_static_pct.to_numpy())
    raw_mean, raw_ci = mean_ci(
        seed_level.gain_vs_unregularized_pct.to_numpy()
    )
    return pd.DataFrame(
        [
            {
                "chosen_alpha": chosen,
                "cells_per_seed": len(DWELLS) * len(DELAYS),
                "independent_seeds": len(seed_level),
                "chosen_skill_vs_static_pct": static_mean,
                "chosen_skill_vs_static_pct_ci95": static_ci,
                "chosen_gain_vs_unregularized_pct": raw_mean,
                "chosen_gain_vs_unregularized_pct_ci95": raw_ci,
            }
        ]
    )


def write_summary(
    chosen: float,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    overall: pd.DataFrame,
) -> None:
    lines = [
        "# Geometry-anchored Bayesian policy selection",
        "",
        f"Validation selected alpha={chosen:g}. The static endpoint alpha=0 was eligible, so the protocol could reject learned decision beliefs.",
        "",
        "## Validation (seeds 0--9, all nine cells)",
        "",
        "Each seed is first averaged over the nine dwell-delay cells; the confidence interval is then formed across seeds.",
        "",
        "| alpha | mean W-AoI skill vs alpha=0 | 95% CI | seeds |",
        "|---:|---:|---:|---:|",
    ]
    for row in validation.itertuples(index=False):
        lines.append(
            f"| {row.alpha:g} | {row.mean_wAoI_skill_vs_alpha0_pct:.3f}% | "
            f"±{row.ci95:.3f}% | {row.validation_seeds} |"
        )
    aggregate = overall.iloc[0]
    lines.extend(
        [
            "",
            "## Frozen test (seeds 20--39)",
            "",
            "Across-grid inference first averages the nine cells within each seed. "
            f"The selected policy changes W-AoI by {aggregate.chosen_skill_vs_static_pct:.2f}% ± "
            f"{aggregate.chosen_skill_vs_static_pct_ci95:.2f}% relative to static geometry and improves it by "
            f"{aggregate.chosen_gain_vs_unregularized_pct:.2f}% ± "
            f"{aggregate.chosen_gain_vs_unregularized_pct_ci95:.2f}% relative to unregularized Bayesian decisions.",
            "",
            "| dwell | delay | skill vs static | gain vs unregularized Bayesian |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in test.itertuples(index=False):
        lines.append(
            f"| {row.dwell:.0f} | {row.delay:.0f} | "
            f"{row.chosen_skill_vs_static_pct:.2f}% ± "
            f"{row.chosen_skill_vs_static_pct_ci95:.2f}% | "
            f"{row.chosen_gain_vs_unregularized_pct:.2f}% ± "
            f"{row.chosen_gain_vs_unregularized_pct_ci95:.2f}% |"
        )
    lines.append("")
    (RESULTS / "regularized_policy_selection_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    validation_tasks = [
        ("validation", alpha, dwell, delay, seed)
        for alpha in ALPHAS
        for dwell in DWELLS
        for delay in DELAYS
        for seed in VALIDATION_SEEDS
    ]
    workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    print(f"Validation: {len(validation_tasks)} runs on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        validation_rows = list(pool.map(run_task, validation_tasks, chunksize=2))
    validation = pd.DataFrame(validation_rows)
    chosen, validation_summary = select_alpha(validation)
    print(f"Selected alpha={chosen:g}", flush=True)

    test_alphas = tuple(dict.fromkeys((0.0, chosen, 1.0)))
    test_tasks = [
        ("test", alpha, dwell, delay, seed)
        for alpha in test_alphas
        for dwell in DWELLS
        for delay in DELAYS
        for seed in TEST_SEEDS
    ]
    print(f"Frozen test: {len(test_tasks)} runs", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        test_rows = list(pool.map(run_task, test_tasks, chunksize=2))
    test = pd.DataFrame(test_rows)
    test_summary = summarize_test(test, chosen)
    overall_test_summary = summarize_overall_test(test, chosen)

    validation.to_csv(RESULTS / "regularized_validation_seeds.csv", index=False)
    validation_summary.to_csv(
        RESULTS / "regularized_validation_summary.csv", index=False
    )
    test.to_csv(RESULTS / "regularized_test_seeds.csv", index=False)
    test_summary.to_csv(RESULTS / "regularized_test_summary.csv", index=False)
    overall_test_summary.to_csv(
        RESULTS / "regularized_test_overall.csv", index=False
    )
    write_summary(chosen, validation_summary, test_summary, overall_test_summary)
    print(validation_summary.to_string(index=False))
    print("\n=== Frozen test ===")
    print(test_summary.to_string(index=False))
    print("\n=== Frozen test, across-grid seed-level inference ===")
    print(overall_test_summary.to_string(index=False))


if __name__ == "__main__":
    main()
