# -*- coding: utf-8 -*-
"""Evidence-dense v3 panels for the FreshSky-Safe TMC manuscript.

Every plotted value is calculated from an executed seed-level CSV.  The
module deliberately keeps inferential units visible and exports every
derived quantity used in a panel.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm


MM = 1.0 / 25.4
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
BLACK = "#242424"
DARK_GREY = "#666666"
MID_GREY = "#9B9B9B"
LIGHT_GREY = "#D9D9D9"
PALE_GREY = "#F2F2F2"

T_CRITICAL = {
    5: 2.7764451051977987,
    10: 2.2621571627409915,
    20: 2.093024054408263,
}


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def _mean_ci(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, float("nan")
    tcrit = T_CRITICAL.get(len(values), 1.96)
    half = tcrit * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return mean, half


def _panel(ax, letter: str, x=-0.12, y=1.08):
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _grid(ax, axis="both"):
    ax.grid(axis=axis, color="#E9E9E9", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(target.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        target.with_suffix(".png"),
        dpi=450,
        bbox_inches="tight",
        pad_inches=0.02,
    )


def _heat_text_color(image, value: float) -> str:
    rgba = image.cmap(image.norm(value))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if luminance < 0.48 else BLACK


def robustness_figure_v3(
    calibration_seed: pd.DataFrame,
    measured: pd.DataFrame,
    derived: Path,
    outdir: Path,
):
    """Fig. 3: estimator calibration, approximation gap, and transfer."""
    conservative = "Conservative-AoI Bayes-Safe"
    expected = "Expected-AoI Bayes-Safe"
    oracle = "Perfect-AoI Bayes-Safe"

    cal_rows = []
    for (dwell, delay), part in calibration_seed.groupby(
        ["mean_dwell", "extra_delay"]
    ):
        pivot = part.pivot(index="seed", columns="policy", values="wAoI")
        paired = pivot[[conservative, expected, oracle]].dropna()
        for seed_id, row in paired.iterrows():
            cal_rows.append(
                {
                    "mean_dwell": float(dwell),
                    "extra_delay": int(delay),
                    "seed": int(seed_id),
                    "gain_vs_conservative_pct": 100.0
                    * (row[conservative] - row[expected])
                    / row[conservative],
                    "gap_vs_true_aoi_pct": 100.0
                    * (row[expected] - row[oracle])
                    / row[oracle],
                }
            )
    cal_effect = pd.DataFrame(cal_rows)
    cal_effect.to_csv(
        derived / "tmc_v3_calibration_effects_seeds.csv", index=False
    )

    cal_summary_rows = []
    for (dwell, delay), part in cal_effect.groupby(
        ["mean_dwell", "extra_delay"]
    ):
        gain, gain_ci = _mean_ci(part["gain_vs_conservative_pct"])
        gap, gap_ci = _mean_ci(part["gap_vs_true_aoi_pct"])
        cal_summary_rows.append(
            {
                "mean_dwell": dwell,
                "extra_delay": delay,
                "gain_vs_conservative_pct": gain,
                "gain_ci95": gain_ci,
                "gap_vs_true_aoi_pct": gap,
                "gap_ci95": gap_ci,
                "paired_seeds": part["seed"].nunique(),
            }
        )
    cal_summary = pd.DataFrame(cal_summary_rows)
    cal_summary.to_csv(
        derived / "tmc_v3_calibration_effects_summary.csv", index=False
    )

    measured_metrics = ["wAoI", "p99_slot_aoi", "cvar95_slot_aoi"]
    measured_rows = []
    relation_rows = []
    for (fraction, delay), part in measured.groupby(
        ["good_fraction", "extra_delay"]
    ):
        metric_pivots = {
            metric: part.pivot(
                index="seed", columns="policy", values=metric
            )
            for metric in measured_metrics + ["age_mae"]
        }
        valid = metric_pivots["wAoI"][
            [conservative, expected, oracle]
        ].dropna().index
        for metric in measured_metrics:
            pivot = metric_pivots[metric].loc[valid]
            for seed_id in valid:
                denom = float(pivot.loc[seed_id, conservative])
                gain = (
                    100.0
                    * (
                        pivot.loc[seed_id, conservative]
                        - pivot.loc[seed_id, expected]
                    )
                    / denom
                    if denom != 0
                    else 0.0
                )
                measured_rows.append(
                    {
                        "good_fraction": float(fraction),
                        "extra_delay": int(delay),
                        "seed": int(seed_id),
                        "metric": metric,
                        "effect_type": "gain_vs_conservative_pct",
                        "effect": float(gain),
                    }
                )
        w_pivot = metric_pivots["wAoI"].loc[valid]
        mae_pivot = metric_pivots["age_mae"].loc[valid]
        for seed_id in valid:
            oracle_gap = (
                100.0
                * (
                    w_pivot.loc[seed_id, expected]
                    - w_pivot.loc[seed_id, oracle]
                )
                / w_pivot.loc[seed_id, oracle]
            )
            measured_rows.append(
                {
                    "good_fraction": float(fraction),
                    "extra_delay": int(delay),
                    "seed": int(seed_id),
                    "metric": "oracle_gap",
                    "effect_type": "expected_gap_vs_oracle_pct",
                    "effect": float(oracle_gap),
                }
            )
            aoi_gain = (
                100.0
                * (
                    w_pivot.loc[seed_id, conservative]
                    - w_pivot.loc[seed_id, expected]
                )
                / w_pivot.loc[seed_id, conservative]
            )
            mae_reduction = (
                100.0
                * (
                    mae_pivot.loc[seed_id, conservative]
                    - mae_pivot.loc[seed_id, expected]
                )
                / mae_pivot.loc[seed_id, conservative]
            )
            relation_rows.append(
                {
                    "good_fraction": float(fraction),
                    "extra_delay": int(delay),
                    "seed": int(seed_id),
                    "age_mae_reduction_pct": float(mae_reduction),
                    "wAoI_gain_pct": float(aoi_gain),
                }
            )

    measured_long = pd.DataFrame(measured_rows)
    measured_relation = pd.DataFrame(relation_rows)
    measured_long.to_csv(
        derived / "tmc_v3_measured_multimetric_effects_seeds.csv",
        index=False,
    )
    measured_relation.to_csv(
        derived / "tmc_v3_measured_relation_seeds.csv", index=False
    )

    # Preserve the v2 contract used by the evidence runner.
    measured_relation[
        ["good_fraction", "extra_delay", "seed", "wAoI_gain_pct"]
    ].rename(columns={"wAoI_gain_pct": "gain_vs_conservative_pct"}).to_csv(
        derived / "tmc_advanced_measured_effects_seeds.csv", index=False
    )

    measured_summary_rows = []
    for keys, part in measured_long.groupby(
        ["good_fraction", "extra_delay", "metric", "effect_type"]
    ):
        value, half = _mean_ci(part["effect"])
        measured_summary_rows.append(
            {
                "good_fraction": keys[0],
                "extra_delay": keys[1],
                "metric": keys[2],
                "effect_type": keys[3],
                "mean_effect": value,
                "ci95": half,
                "paired_seeds": part["seed"].nunique(),
            }
        )
    measured_summary = pd.DataFrame(measured_summary_rows)
    measured_summary.to_csv(
        derived / "tmc_v3_measured_multimetric_effects_summary.csv",
        index=False,
    )

    fig = plt.figure(figsize=(183 * MM, 91 * MM))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.92, 1.08],
        width_ratios=[1.03, 1.0],
        hspace=0.58,
        wspace=0.42,
    )
    ax_gain = fig.add_subplot(gs[0, 0])
    ax_gap = fig.add_subplot(gs[0, 1])
    ax_transfer = fig.add_subplot(gs[1, 0])
    ax_relation = fig.add_subplot(gs[1, 1])

    dwell = sorted(cal_summary["mean_dwell"].unique())
    delays = sorted(cal_summary["extra_delay"].unique())

    def calibration_matrix(field):
        return (
            cal_summary.pivot(
                index="mean_dwell", columns="extra_delay", values=field
            )
            .reindex(index=dwell, columns=delays)
            .to_numpy()
        )

    gain = calibration_matrix("gain_vs_conservative_pct")
    gain_ci = calibration_matrix("gain_ci95")
    image_gain = ax_gain.imshow(
        gain,
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=max(42.0, float(np.nanmax(gain))),
    )
    for i in range(len(dwell)):
        for j in range(len(delays)):
            ax_gain.text(
                j,
                i,
                f"{gain[i, j]:.1f}\n±{gain_ci[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=5.8,
                linespacing=0.9,
                color=_heat_text_color(image_gain, gain[i, j]),
            )
    ax_gain.set_title("Gain over conservative estimator (%)", pad=3)
    ax_gain.set_xticks(range(len(delays)), [f"$d={d}$" for d in delays])
    ax_gain.set_yticks(range(len(dwell)), [f"{x:g}" for x in dwell])
    ax_gain.set_xlabel("pending-feedback delay")
    ax_gain.set_ylabel("mean NLoS dwell (slots)")
    for spine in ax_gain.spines.values():
        spine.set_visible(False)
    _panel(ax_gain, "a", x=-0.21, y=1.16)

    gap = calibration_matrix("gap_vs_true_aoi_pct")
    gap_ci = calibration_matrix("gap_ci95")
    image_gap = ax_gap.imshow(
        gap,
        aspect="auto",
        cmap="YlOrBr",
        vmin=0,
        vmax=max(20.0, float(np.nanmax(gap))),
    )
    for i in range(len(dwell)):
        for j in range(len(delays)):
            ax_gap.text(
                j,
                i,
                f"{gap[i, j]:.1f}\n±{gap_ci[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=5.8,
                linespacing=0.9,
                color=_heat_text_color(image_gap, gap[i, j]),
            )
            if gap[i, j] > 5.0:
                ax_gap.plot(
                    j + 0.32,
                    i - 0.32,
                    marker="^",
                    markersize=2.8,
                    color=VERMILLION,
                    markeredgecolor="white",
                    markeredgewidth=0.25,
                )
    ax_gap.set_title("Gap to perfect receiver age (%)", pad=3)
    ax_gap.set_xticks(range(len(delays)), [f"$d={d}$" for d in delays])
    ax_gap.set_yticks(range(len(dwell)), [f"{x:g}" for x in dwell])
    ax_gap.set_xlabel("pending-feedback delay")
    ax_gap.set_ylabel("mean NLoS dwell (slots)")
    ax_gap.text(
        0.99,
        -0.27,
        "▲ gap > 5%",
        transform=ax_gap.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        color=VERMILLION,
    )
    for spine in ax_gap.spines.values():
        spine.set_visible(False)
    _panel(ax_gap, "b", x=-0.21, y=1.16)

    fractions = sorted(measured_summary["good_fraction"].unique())
    conditions = [(f, d) for f in fractions for d in [4, 8]]
    metric_order = ["wAoI", "p99_slot_aoi", "cvar95_slot_aoi", "oracle_gap"]
    metric_labels = ["W-AoI", "p99 AoI", "CVaR95", "oracle gap"]
    matrix = np.empty((len(conditions), len(metric_order)))
    ci_matrix = np.empty_like(matrix)
    for i, (fraction, delay) in enumerate(conditions):
        for j, metric in enumerate(metric_order):
            row = measured_summary[
                np.isclose(measured_summary["good_fraction"], fraction)
                & (measured_summary["extra_delay"] == delay)
                & (measured_summary["metric"] == metric)
            ].iloc[0]
            # Benefit columns are positive when expected age helps.  The
            # oracle-gap column is negated for a common color semantics.
            sign = -1.0 if metric == "oracle_gap" else 1.0
            matrix[i, j] = sign * float(row["mean_effect"])
            ci_matrix[i, j] = float(row["ci95"])
    limit = max(42.0, float(np.nanmax(np.abs(matrix))))
    transfer_image = ax_transfer.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            shown = -matrix[i, j] if metric_order[j] == "oracle_gap" else matrix[i, j]
            ax_transfer.text(
                j,
                i,
                f"{shown:.1f}\n±{ci_matrix[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=5.2,
                linespacing=0.9,
                color=_heat_text_color(transfer_image, matrix[i, j]),
            )
    ax_transfer.set_xticks(range(4), metric_labels)
    ax_transfer.tick_params(axis="x", rotation=23)
    ax_transfer.set_yticks(
        range(len(conditions)),
        [
            f"$q={0.25 * (fractions.index(f) + 1):.2f}$, $d={d}$"
            for f, d in conditions
        ],
    )
    ax_transfer.set_title(
        "Measured-trace transfer: gain vs. conservative; oracle gap (%)",
        pad=3,
    )
    for spine in ax_transfer.spines.values():
        spine.set_visible(False)
    _panel(ax_transfer, "c", x=-0.21, y=1.13)

    delay_marker = {4: "o", 8: "s"}
    fraction_color = {
        fractions[0]: VERMILLION,
        fractions[1]: ORANGE,
        fractions[2]: BLUE,
    }
    for (fraction, delay), part in measured_relation.groupby(
        ["good_fraction", "extra_delay"]
    ):
        ax_relation.scatter(
            part["age_mae_reduction_pct"],
            part["wAoI_gain_pct"],
            s=11,
            marker=delay_marker[int(delay)],
            facecolor=fraction_color[fraction],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.62,
            zorder=2,
        )
        cx = float(part["age_mae_reduction_pct"].mean())
        cy = float(part["wAoI_gain_pct"].mean())
        ax_relation.scatter(
            cx,
            cy,
            s=34,
            marker=delay_marker[int(delay)],
            facecolor=fraction_color[fraction],
            edgecolor=BLACK,
            linewidth=0.65,
            zorder=3,
        )
    x = measured_relation["age_mae_reduction_pct"].to_numpy()
    y = measured_relation["wAoI_gain_pct"].to_numpy()
    corr = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.linspace(float(x.min()), float(x.max()), 100)
    ax_relation.plot(
        line_x,
        slope * line_x + intercept,
        color=DARK_GREY,
        linewidth=0.9,
        linestyle="--",
        zorder=1,
    )
    ax_relation.axvline(0, color=LIGHT_GREY, linewidth=0.65)
    ax_relation.text(
        0.03,
        0.97,
        f"seed-level descriptive $r={corr:.2f}$\n$n={len(measured_relation)}$ pairs",
        transform=ax_relation.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
    )
    for fraction, color in fraction_color.items():
        quantile = fractions.index(fraction) + 1
        ax_relation.scatter(
            [], [],
            s=19,
            color=color,
            label=f"$q={0.25*quantile:.2f}$",
        )
    for delay, marker in delay_marker.items():
        ax_relation.scatter(
            [], [],
            s=19,
            marker=marker,
            facecolor="none",
            edgecolor=BLACK,
            label=f"$d={delay}$",
        )
    ax_relation.set_xlabel("receiver-age MAE reduction (%)")
    ax_relation.set_ylabel("W-AoI gain over conservative (%)")
    ax_relation.legend(
        loc="lower right",
        ncol=2,
        fontsize=5.2,
        handletextpad=0.25,
        columnspacing=0.7,
    )
    _grid(ax_relation)
    _panel(ax_relation, "d", x=-0.09, y=1.13)

    _save(fig, outdir / "fig_tmc_robustness")
    plt.close(fig)


def _factorial_codes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["adaptive_belief"] = _as_bool(frame["adaptive_belief"])
    frame["shield"] = _as_bool(frame["shield"])
    frame["A"] = np.where(frame["adaptive_belief"], 1.0, -1.0)
    frame["E"] = np.where(
        frame["age_estimator"].eq("expected"), 1.0, -1.0
    )
    frame["S"] = np.where(frame["shield"], 1.0, -1.0)
    return frame


def factorial_figure_v3(
    factorial: pd.DataFrame, derived: Path, outdir: Path
):
    """Fig. 5: raw factorial cells, multi-outcome effects, shield trade-off."""
    frame = _factorial_codes(factorial)
    terms = {
        "Expected age": ("E",),
        "Bayesian belief": ("A",),
        "Safety shield": ("S",),
        "Belief × age": ("A", "E"),
        "Belief × shield": ("A", "S"),
        "Age × shield": ("E", "S"),
        "Three-way": ("A", "E", "S"),
    }
    metrics = {
        "wAoI": ("W-AoI", "relative_pct"),
        "p99_slot_aoi": ("p99 AoI", "relative_pct"),
        "worst_user_mean_aoi": ("worst user", "relative_pct"),
        "max_avg_power": ("max power", "relative_pct"),
        "rejection_rate": ("rejection", "percentage_points"),
    }

    effect_rows = []
    for seed_id, part in frame.groupby("seed"):
        if len(part) != 8:
            continue
        for effect, codes in terms.items():
            sign = np.ones(len(part))
            for code in codes:
                sign *= part[code].to_numpy()
            for metric, (label, unit) in metrics.items():
                contrast = float(np.sum(sign * part[metric]) / 4.0)
                if unit == "relative_pct":
                    value = 100.0 * contrast / float(part[metric].mean())
                else:
                    value = 100.0 * contrast
                effect_rows.append(
                    {
                        "seed": int(seed_id),
                        "effect": effect,
                        "metric": metric,
                        "metric_label": label,
                        "unit": unit,
                        "effect_value": value,
                    }
                )
    effect_long = pd.DataFrame(effect_rows)
    effect_long.to_csv(
        derived / "tmc_v3_factorial_multimetric_effects_seeds.csv",
        index=False,
    )

    summary_rows = []
    for keys, part in effect_long.groupby(
        ["effect", "metric", "metric_label", "unit"]
    ):
        mean, half = _mean_ci(part["effect_value"])
        summary_rows.append(
            {
                "effect": keys[0],
                "metric": keys[1],
                "metric_label": keys[2],
                "unit": keys[3],
                "mean_effect": mean,
                "ci95": half,
                "seeds": part["seed"].nunique(),
            }
        )
    effect_summary = pd.DataFrame(summary_rows)
    effect_summary.to_csv(
        derived / "tmc_v3_factorial_multimetric_effects_summary.csv",
        index=False,
    )

    # Preserve the v2 single-outcome evidence tables.
    legacy_seed = effect_long[effect_long["metric"] == "wAoI"][
        ["seed", "effect", "effect_value"]
    ].rename(columns={"effect_value": "wAoI_effect_pct"})
    legacy_seed["wAoI_effect"] = np.nan
    # Recover the unnormalised contrast exactly for the old contract.
    raw_rows = []
    for seed_id, part in frame.groupby("seed"):
        grand = float(part["wAoI"].mean())
        for effect, codes in terms.items():
            sign = np.ones(len(part))
            for code in codes:
                sign *= part[code].to_numpy()
            contrast = float(np.sum(sign * part["wAoI"]) / 4.0)
            raw_rows.append(
                {
                    "seed": int(seed_id),
                    "effect": effect,
                    "wAoI_effect": contrast,
                    "wAoI_effect_pct": 100.0 * contrast / grand,
                }
            )
    legacy_seed = pd.DataFrame(raw_rows)
    legacy_seed.to_csv(
        derived / "tmc_advanced_factorial_contrasts_seeds.csv", index=False
    )
    legacy_summary_rows = []
    for effect, part in legacy_seed.groupby("effect"):
        mean, half = _mean_ci(part["wAoI_effect_pct"])
        legacy_summary_rows.append(
            {
                "effect": effect,
                "mean_effect_pct": mean,
                "ci95_halfwidth": half,
                "seeds": part["seed"].nunique(),
            }
        )
    pd.DataFrame(legacy_summary_rows).to_csv(
        derived / "tmc_advanced_factorial_contrasts_summary.csv",
        index=False,
    )

    trade_rows = []
    labels = {
        (False, "conservative"): "Static / cons.",
        (False, "expected"): "Static / exp.",
        (True, "conservative"): "Bayes / cons.",
        (True, "expected"): "Bayes / exp.",
    }
    for key, label in labels.items():
        adaptive, age_mode = key
        part = frame[
            (frame["adaptive_belief"] == adaptive)
            & (frame["age_estimator"] == age_mode)
        ]
        aoi = part.pivot(index="seed", columns="shield", values="wAoI")
        power = part.pivot(
            index="seed", columns="shield", values="max_avg_power"
        )
        valid = aoi.index.intersection(power.index)
        for seed_id in valid:
            trade_rows.append(
                {
                    "adaptive_belief": adaptive,
                    "age_estimator": age_mode,
                    "label": label,
                    "seed": int(seed_id),
                    "aoi_cost_pct": 100.0
                    * (aoi.loc[seed_id, True] - aoi.loc[seed_id, False])
                    / aoi.loc[seed_id, False],
                    "power_reduction_pct": 100.0
                    * (
                        power.loc[seed_id, False]
                        - power.loc[seed_id, True]
                    )
                    / power.loc[seed_id, False],
                }
            )
    trade_seed = pd.DataFrame(trade_rows)
    trade_seed.to_csv(
        derived / "tmc_v3_shield_tradeoff_seeds.csv", index=False
    )
    trade_summary_rows = []
    for keys, part in trade_seed.groupby(
        ["adaptive_belief", "age_estimator", "label"]
    ):
        x, xci = _mean_ci(part["aoi_cost_pct"])
        y, yci = _mean_ci(part["power_reduction_pct"])
        trade_summary_rows.append(
            {
                "adaptive_belief": keys[0],
                "age_estimator": keys[1],
                "label": keys[2],
                "aoi_cost_pct": x,
                "aoi_cost_ci95": xci,
                "power_reduction_pct": y,
                "power_reduction_ci95": yci,
                "paired_seeds": part["seed"].nunique(),
            }
        )
    trade_summary = pd.DataFrame(trade_summary_rows)
    trade_summary.to_csv(
        derived / "tmc_advanced_shield_tradeoff.csv", index=False
    )

    fig = plt.figure(figsize=(183 * MM, 78 * MM))
    outer = fig.add_gridspec(
        1, 3, width_ratios=[1.03, 1.36, 1.03], wspace=0.42
    )
    ax_raw = fig.add_subplot(outer[0, 0])
    effect_grid = outer[0, 1].subgridspec(
        1, 2, width_ratios=[0.82, 0.18], wspace=0.07
    )
    ax_effect = fig.add_subplot(effect_grid[0, 0])
    ax_reject = fig.add_subplot(effect_grid[0, 1], sharey=ax_effect)
    ax_trade = fig.add_subplot(outer[0, 2])

    row_keys = list(labels)
    rng = np.random.default_rng(20260730)
    for yy, key in enumerate(row_keys):
        adaptive, age_mode = key
        part = frame[
            (frame["adaptive_belief"] == adaptive)
            & (frame["age_estimator"] == age_mode)
        ]
        pivot = part.pivot(index="seed", columns="shield", values="wAoI")
        jitter = rng.uniform(-0.10, 0.10, size=len(pivot))
        for offset, (_, row) in zip(jitter, pivot.iterrows()):
            ax_raw.plot(
                [row[False], row[True]],
                [yy + offset, yy + offset],
                color="#C7C7C7",
                linewidth=0.45,
                alpha=0.65,
                zorder=1,
            )
        ax_raw.scatter(
            pivot[False],
            yy + jitter,
            s=7,
            color=MID_GREY,
            alpha=0.55,
            linewidth=0,
            zorder=2,
        )
        ax_raw.scatter(
            pivot[True],
            yy + jitter,
            s=9,
            marker="^",
            color=GREEN,
            alpha=0.58,
            linewidth=0,
            zorder=2,
        )
        off_mean, off_ci = _mean_ci(pivot[False])
        on_mean, on_ci = _mean_ci(pivot[True])
        ax_raw.errorbar(
            [off_mean, on_mean],
            [yy - 0.18, yy + 0.18],
            xerr=[off_ci, on_ci],
            fmt="none",
            ecolor=BLACK,
            capsize=1.5,
            elinewidth=0.8,
            zorder=3,
        )
        ax_raw.scatter(
            off_mean,
            yy - 0.18,
            s=24,
            color=MID_GREY,
            edgecolor=BLACK,
            linewidth=0.5,
            zorder=4,
        )
        ax_raw.scatter(
            on_mean,
            yy + 0.18,
            s=28,
            marker="^",
            color=GREEN,
            edgecolor=BLACK,
            linewidth=0.5,
            zorder=4,
        )
    ax_raw.set_yticks(range(4), [labels[key] for key in row_keys])
    ax_raw.invert_yaxis()
    ax_raw.set_xlabel("W-AoI (slots)")
    ax_raw.scatter(
        [], [], s=18, color=MID_GREY, label="shield off"
    )
    ax_raw.scatter(
        [], [], s=20, marker="^", color=GREEN, label="shield on"
    )
    ax_raw.legend(
        loc="lower right",
        fontsize=5.3,
        handletextpad=0.25,
        labelspacing=0.2,
    )
    _grid(ax_raw, "x")
    _panel(ax_raw, "a", x=-0.26, y=1.07)

    order = list(terms)
    metric_order = [
        "wAoI",
        "p99_slot_aoi",
        "worst_user_mean_aoi",
        "max_avg_power",
    ]
    matrix = np.empty((len(order), len(metric_order)))
    ci_matrix = np.empty_like(matrix)
    for i, effect in enumerate(order):
        for j, metric in enumerate(metric_order):
            row = effect_summary[
                (effect_summary["effect"] == effect)
                & (effect_summary["metric"] == metric)
            ].iloc[0]
            matrix[i, j] = float(row["mean_effect"])
            ci_matrix[i, j] = float(row["ci95"])
    limit = max(36.0, float(np.nanmax(np.abs(matrix))))
    image = ax_effect.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax_effect.text(
                j,
                i,
                f"{matrix[i, j]:+.1f}",
                ha="center",
                va="center",
                fontsize=5.2,
                color=_heat_text_color(image, matrix[i, j]),
            )
            if abs(matrix[i, j]) > ci_matrix[i, j]:
                ax_effect.plot(
                    j + 0.34,
                    i - 0.33,
                    marker="o",
                    markersize=1.6,
                    color=BLACK,
                    markeredgecolor="white",
                    markeredgewidth=0.2,
                )
    ax_effect.set_xticks(
        range(4), ["W-AoI", "p99", "worst\nuser", "max\npower"]
    )
    ax_effect.set_yticks(range(len(order)), order)
    ax_effect.set_title("factorial contrast (% of grand mean)", pad=3)
    for spine in ax_effect.spines.values():
        spine.set_visible(False)
    _panel(ax_effect, "b", x=-0.41, y=1.07)

    reject = np.empty((len(order), 1))
    reject_ci = np.empty_like(reject)
    for i, effect in enumerate(order):
        row = effect_summary[
            (effect_summary["effect"] == effect)
            & (effect_summary["metric"] == "rejection_rate")
        ].iloc[0]
        reject[i, 0] = float(row["mean_effect"])
        reject_ci[i, 0] = float(row["ci95"])
    reject_limit = max(5.0, float(np.nanmax(np.abs(reject))))
    reject_image = ax_reject.imshow(
        reject,
        aspect="auto",
        cmap="PuOr_r",
        norm=TwoSlopeNorm(
            vmin=-reject_limit, vcenter=0.0, vmax=reject_limit
        ),
    )
    for i in range(len(order)):
        ax_reject.text(
            0,
            i,
            f"{reject[i, 0]:+.1f}",
            ha="center",
            va="center",
            fontsize=5.1,
            color=_heat_text_color(reject_image, reject[i, 0]),
        )
        if abs(reject[i, 0]) > reject_ci[i, 0]:
            ax_reject.plot(
                0.32,
                i - 0.33,
                marker="o",
                markersize=1.6,
                color=BLACK,
                markeredgecolor="white",
                markeredgewidth=0.2,
            )
    ax_reject.set_xticks([0], ["reject.\n(pp)"])
    ax_reject.tick_params(axis="y", left=False, labelleft=False)
    ax_reject.set_title(" ", pad=3)
    for spine in ax_reject.spines.values():
        spine.set_visible(False)
    ax_effect.text(
        0.0,
        -0.19,
        "● 95% CI excludes zero; negative lowers an outcome",
        transform=ax_effect.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
    )

    config_color = {
        "Static / cons.": MID_GREY,
        "Static / exp.": GREEN,
        "Bayes / cons.": PURPLE,
        "Bayes / exp.": BLUE,
    }
    config_marker = {
        "Static / cons.": "o",
        "Static / exp.": "s",
        "Bayes / cons.": "D",
        "Bayes / exp.": "^",
    }
    for label, part in trade_seed.groupby("label"):
        ax_trade.scatter(
            part["aoi_cost_pct"],
            part["power_reduction_pct"],
            s=8,
            marker=config_marker[label],
            color=config_color[label],
            alpha=0.30,
            linewidth=0,
            zorder=1,
        )
        row = trade_summary[trade_summary["label"] == label].iloc[0]
        ax_trade.errorbar(
            row["aoi_cost_pct"],
            row["power_reduction_pct"],
            xerr=row["aoi_cost_ci95"],
            yerr=row["power_reduction_ci95"],
            fmt=config_marker[label],
            markersize=4.5,
            color=config_color[label],
            markeredgecolor=BLACK,
            markeredgewidth=0.45,
            capsize=1.7,
            elinewidth=0.75,
            label=label,
            zorder=3,
        )
    ax_trade.axhline(0, color=LIGHT_GREY, linewidth=0.6)
    ax_trade.axvline(0, color=LIGHT_GREY, linewidth=0.6)
    ax_trade.set_xlabel("W-AoI cost of shield (%)")
    ax_trade.set_ylabel("max-power reduction (%)")
    ax_trade.legend(
        loc="upper right",
        fontsize=5.0,
        handletextpad=0.25,
        labelspacing=0.2,
    )
    _grid(ax_trade)
    _panel(ax_trade, "c", x=-0.22, y=1.07)

    _save(fig, outdir / "fig_tmc_factorial_effects")
    plt.close(fig)


def runtime_figure_v3(runtime: pd.DataFrame, derived: Path, outdir: Path):
    """Fig. 7: full 54-point latency cube, bottleneck, and memory."""
    frame = runtime.copy()
    frame["total_p95_ms"] = frame["total_p95_us"] / 1000.0
    component_fields = [
        "expected_age_median_us",
        "decision_median_us",
        "accounting_median_us",
        "filter_update_median_us",
    ]
    for field in component_fields:
        frame[field.replace("_us", "_ms")] = frame[field] / 1000.0
    frame.to_csv(derived / "tmc_v3_runtime_grid.csv", index=False)

    focus = frame[
        (frame["retained_models"] == 144)
        & (frame["extra_delay"] == 8)
    ].sort_values("N")
    focus.to_csv(derived / "tmc_advanced_runtime_focus.csv", index=False)

    fig = plt.figure(figsize=(183 * MM, 76 * MM))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.45, 1.02, 0.92], wspace=0.47
    )
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_stack = fig.add_subplot(gs[0, 1])
    ax_memory = fig.add_subplot(gs[0, 2])

    n_order = sorted(frame["N"].unique())
    columns = [
        (delay, retained)
        for delay in sorted(frame["extra_delay"].unique())
        for retained in sorted(frame["retained_models"].unique())
    ]
    values = np.empty((len(n_order), len(columns)))
    for i, n in enumerate(n_order):
        for j, (delay, retained) in enumerate(columns):
            values[i, j] = frame[
                (frame["N"] == n)
                & (frame["extra_delay"] == delay)
                & (frame["retained_models"] == retained)
            ]["total_p95_ms"].iloc[0]
    image = ax_heat.imshow(
        values,
        aspect="auto",
        cmap="YlGnBu",
        norm=LogNorm(
            vmin=max(float(np.nanmin(values)), 1e-3),
            vmax=float(np.nanmax(values)),
        ),
    )
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax_heat.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=4.75,
                color=_heat_text_color(image, values[i, j]),
            )
    ax_heat.set_xticks(
        range(len(columns)), [f"{int(k)}" for _, k in columns]
    )
    ax_heat.set_yticks(
        range(len(n_order)), [f"{int(n)}" for n in n_order]
    )
    ax_heat.set_xlabel("retained channel models $K$")
    ax_heat.set_ylabel("number of UAVs $N$")
    for x in [2.5, 5.5]:
        ax_heat.axvline(x, color="white", linewidth=1.2)
    for start, delay in zip([1.0, 4.0, 7.0], [0, 4, 8]):
        ax_heat.text(
            start,
            -0.88,
            f"$d={delay}$",
            ha="center",
            va="center",
            fontsize=5.8,
            fontweight="bold",
        )
    ax_heat.set_ylim(len(n_order) - 0.5, -1.15)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(
        image, ax=ax_heat, orientation="horizontal", fraction=0.045, pad=0.16
    )
    cbar.set_label("end-to-end p95 decision time (ms)", labelpad=1)
    cbar.ax.tick_params(labelsize=5.2, length=2)
    _panel(ax_heat, "a", x=-0.20, y=1.11)

    y = np.arange(len(focus))
    left = np.zeros(len(focus))
    components = [
        ("expected_age_median_ms", "receiver-age", SKY),
        ("decision_median_ms", "rank + shield", BLUE),
        ("accounting_median_ms", "accounting", GREEN),
        ("filter_update_median_ms", "Bayes filter", ORANGE),
    ]
    for field, label, color in components:
        component = focus[field].to_numpy()
        ax_stack.barh(
            y,
            component,
            left=left,
            height=0.56,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            label=label,
        )
        left += component
    ax_stack.scatter(
        focus["total_p95_ms"],
        y,
        color=BLACK,
        marker="D",
        s=14,
        label="end-to-end p95",
        zorder=4,
    )
    for yy, p95 in zip(y, focus["total_p95_ms"]):
        ax_stack.annotate(
            f"{p95:.2f}",
            (p95, yy),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=4.9,
        )
    ax_stack.set_yticks(y, [f"$N={int(n)}$" for n in focus["N"]])
    ax_stack.invert_yaxis()
    ax_stack.set_xlabel("decision time (ms), $K=144$, $d=8$")
    ax_stack.legend(
        loc="lower center",
        bbox_to_anchor=(0.50, 1.01),
        fontsize=4.8,
        ncol=2,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.6,
        labelspacing=0.2,
    )
    _grid(ax_stack, "x")
    _panel(ax_stack, "b", x=-0.25, y=1.11)

    k_colors = {36: SKY, 144: BLUE, 400: VERMILLION}
    for retained in sorted(frame["retained_models"].unique()):
        part = frame[frame["retained_models"] == retained]
        memory = part.pivot(
            index="N", columns="extra_delay", values="total_memory_kib"
        ).sort_index()
        x = memory.index.to_numpy()
        lower = memory.min(axis=1).to_numpy()
        upper = memory.max(axis=1).to_numpy()
        mid = memory[4].to_numpy()
        color = k_colors[int(retained)]
        ax_memory.fill_between(
            x, lower, upper, color=color, alpha=0.13, linewidth=0
        )
        ax_memory.plot(
            x,
            mid,
            marker="o",
            markersize=2.8,
            linewidth=1.0,
            color=color,
        )
        ax_memory.annotate(
            f"$K={int(retained)}$",
            (x[-1], mid[-1]),
            xytext=(3, 0),
            textcoords="offset points",
            fontsize=5.2,
            color=color,
            va="center",
        )
    max_row = frame.loc[frame["total_memory_kib"].idxmax()]
    ax_memory.scatter(
        max_row["N"],
        max_row["total_memory_kib"],
        marker="D",
        s=18,
        color=BLACK,
        zorder=4,
    )
    ax_memory.annotate(
        f"{max_row['total_memory_kib']/1024:.2f} MiB",
        (max_row["N"], max_row["total_memory_kib"]),
        xytext=(-5, 7),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=5.2,
    )
    ax_memory.axhline(
        1024, color=DARK_GREY, linestyle=":", linewidth=0.7
    )
    ax_memory.set_xscale("log")
    ax_memory.set_yscale("log")
    ax_memory.set_xticks([12, 50, 200, 500], ["12", "50", "200", "500"])
    ax_memory.set_xlabel("number of UAVs $N$")
    ax_memory.set_ylabel("total state memory (KiB)")
    ax_memory.text(
        0.03,
        0.97,
        "line: $d=4$; band: $d=0$–8",
        transform=ax_memory.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
    )
    _grid(ax_memory)
    _panel(ax_memory, "c", x=-0.25, y=1.11)

    _save(fig, outdir / "fig_tmc_runtime")
    plt.close(fig)
