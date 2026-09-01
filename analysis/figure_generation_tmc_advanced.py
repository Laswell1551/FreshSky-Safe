# -*- coding: utf-8 -*-
"""High-information-density, journal-only figures for FreshSky-Safe.

The script reads executed seed-level CSV files.  It never fabricates or
digitizes values.  In addition to vector figures, it exports the derived
paired effects, factorial contrasts, Qmax summaries, and per-UAV long table.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from figure_panels_v3 import (
    factorial_figure_v3,
    robustness_figure_v3,
    runtime_figure_v3,
)
from figure_service_v3 import make_figure as service_figure_v3
from figure_baseline_v3 import make_figure as baseline_figure_v3


MM = 1.0 / 25.4
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
YELLOW = "#F0E442"
BLACK = "#242424"
DARK_GREY = "#666666"
MID_GREY = "#9B9B9B"
LIGHT_GREY = "#D9D9D9"
PALE_GREY = "#F2F2F2"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
        "font.size": 7.2,
        "axes.titlesize": 7.4,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 6.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.7,
        "ytick.major.size": 2.7,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.dpi": 450,
    }
)

SHORT = {
    "FreshSky-Safe": "FreshSky",
    "Ji'24-style Greedy-Safe": "Ji'24 greedy",
    "Zhao'25 MMSE-MW-Safe": "Zhao'25 MW-E",
    "Zhu'26-style Aggregate-DPP": "Zhu'26 agg.-DPP",
    "Known-Transition Safe": "Known transition",
    "Wang'26 PORMAB-Safe": "Wang'26 PORMAB",
    "Tripathi'24 Whittle-Safe": "Tripathi'24 WI",
    "Conservative-AoI Safe": "Conservative age",
    "Static-Geometry Safe": "Static DPP-MC",
    "Max-Age-First Safe": "Max-Age",
    "Round-Robin Safe": "Round-Robin",
}

COLORS = {
    "FreshSky-Safe": BLUE,
    "Ji'24-style Greedy-Safe": SKY,
    "Zhao'25 MMSE-MW-Safe": "#7A5195",
    "Zhu'26-style Aggregate-DPP": VERMILLION,
    "Known-Transition Safe": PURPLE,
    "Wang'26 PORMAB-Safe": ORANGE,
    "Tripathi'24 Whittle-Safe": "#B66D00",
    "Conservative-AoI Safe": MID_GREY,
    "Static-Geometry Safe": GREEN,
    "Max-Age-First Safe": "#5D9BCB",
    "Round-Robin Safe": DARK_GREY,
}

MARKERS = {
    "FreshSky-Safe": "o",
    "Ji'24-style Greedy-Safe": "s",
    "Zhao'25 MMSE-MW-Safe": "p",
    "Zhu'26-style Aggregate-DPP": "X",
    "Known-Transition Safe": "D",
    "Wang'26 PORMAB-Safe": "^",
    "Tripathi'24 Whittle-Safe": "v",
    "Conservative-AoI Safe": "<",
    "Static-Geometry Safe": "P",
    "Max-Age-First Safe": "h",
    "Round-Robin Safe": ">",
}

T_CRITICAL = {
    5: 2.7764451051977987,
    10: 2.2621571627409915,
    20: 2.093024054408263,
}


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def mean_ci(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, float("nan")
    tcrit = T_CRITICAL.get(len(values), 1.96)
    half = tcrit * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return mean, half


def panel(ax, letter: str, x=-0.12, y=1.08):
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


def save(fig, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix(".pdf"))
    fig.savefig(target.with_suffix(".svg"))
    fig.savefig(target.with_suffix(".png"), dpi=450)


def add_grid(ax, axis="both"):
    ax.grid(axis=axis, color="#E9E9E9", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def baseline_effects(seed: pd.DataFrame) -> pd.DataFrame:
    ours = "FreshSky-Safe"
    rows = []
    for delay, part in seed.groupby("extra_delay"):
        pivot = part.pivot(index="seed", columns="policy", values="wAoI")
        for baseline in pivot.columns:
            if baseline == ours:
                continue
            paired = pivot[[ours, baseline]].dropna()
            gain = 100.0 * (
                paired[baseline] - paired[ours]
            ) / paired[baseline]
            mean, half = mean_ci(gain)
            rows.append(
                {
                    "extra_delay": int(delay),
                    "baseline": baseline,
                    "freshsky_gain_pct": mean,
                    "ci95_halfwidth": half,
                    "paired_seeds": len(gain),
                }
            )
    return pd.DataFrame(rows)


def baseline_figure(seed: pd.DataFrame, derived: Path, outdir: Path):
    effects = baseline_effects(seed)
    effects.to_csv(derived / "tmc_advanced_baseline_effects.csv", index=False)

    fig = plt.figure(figsize=(183 * MM, 91 * MM))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.22, 1.0],
        height_ratios=[1.05, 0.95],
        wspace=0.48,
        hspace=0.48,
    )
    ax_forest = fig.add_subplot(gs[:, 0])
    ax_pareto = fig.add_subplot(gs[0, 1])
    ax_rank = fig.add_subplot(gs[1, 1])

    d4 = effects[effects["extra_delay"] == 4].set_index("baseline")
    order = (
        d4["freshsky_gain_pct"]
        .sort_values(ascending=False)
        .index.tolist()
    )
    ybase = np.arange(len(order))
    delay_style = {
        2: (SKY, "o", -0.20),
        4: (BLUE, "s", 0.00),
        8: (ORANGE, "^", 0.20),
    }
    ax_forest.axvspan(-1.0, 1.0, color="#F4F4F4", zorder=0)
    ax_forest.axvline(0, color=BLACK, linewidth=0.75, linestyle="--")
    for delay, (color, marker, offset) in delay_style.items():
        part = effects[effects["extra_delay"] == delay].set_index("baseline")
        vals = part.loc[order]
        ax_forest.errorbar(
            vals["freshsky_gain_pct"],
            ybase + offset,
            xerr=vals["ci95_halfwidth"],
            fmt=marker,
            color=color,
            ecolor=color,
            markersize=3.8,
            elinewidth=0.85,
            capsize=1.8,
            label=f"$d={delay}$",
            zorder=3,
        )
    ax_forest.set_yticks(ybase, [SHORT[p] for p in order])
    ax_forest.invert_yaxis()
    ax_forest.set_xlabel(
        "paired FreshSky W-AoI gain (%)\npositive favors FreshSky"
    )
    ax_forest.legend(
        loc="lower right",
        ncol=3,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    add_grid(ax_forest, "x")
    panel(ax_forest, "a", x=-0.21, y=1.04)

    core = [
        "FreshSky-Safe",
        "Wang'26 PORMAB-Safe",
        "Tripathi'24 Whittle-Safe",
        "Ji'24-style Greedy-Safe",
        "Zhao'25 MMSE-MW-Safe",
        "Static-Geometry Safe",
        "Known-Transition Safe",
        "Zhu'26-style Aggregate-DPP",
    ]
    d4_seed = seed[seed["extra_delay"] == 4]
    ax_pareto.axvspan(0.4, 0.4065, color="#EFEFEF", zorder=0)
    ax_pareto.axvline(0.4, color=BLACK, linestyle="--", linewidth=0.65)
    pareto_rows = []
    for policy in core:
        part = d4_seed[d4_seed["policy"] == policy]
        x, xci = mean_ci(part["max_avg_power"])
        y, yci = mean_ci(part["wAoI"])
        pareto_rows.append(
            {
                "policy": policy,
                "max_avg_power": x,
                "max_avg_power_ci95": xci,
                "wAoI": y,
                "wAoI_ci95": yci,
            }
        )
        face = "none" if policy == "Known-Transition Safe" else COLORS[policy]
        edge = COLORS[policy]
        ax_pareto.errorbar(
            x,
            y,
            xerr=xci,
            yerr=yci,
            fmt="none",
            ecolor=edge,
            elinewidth=0.65,
            capsize=1.5,
            alpha=0.85,
            zorder=2,
        )
        ax_pareto.scatter(
            x,
            y,
            s=30,
            marker=MARKERS[policy],
            facecolors=face,
            edgecolors=edge,
            linewidths=0.9,
            label=SHORT[policy],
            zorder=3,
        )
    pd.DataFrame(pareto_rows).to_csv(
        derived / "tmc_advanced_baseline_pareto.csv", index=False
    )
    ax_pareto.text(
        0.4005,
        0.98,
        "budget",
        transform=ax_pareto.get_xaxis_transform(),
        fontsize=5.4,
        va="top",
    )
    ax_pareto.set_xlabel("maximum per-UAV power (mW)")
    ax_pareto.set_ylabel("W-AoI (slots)")
    ax_pareto.legend(
        loc="upper right",
        ncol=2,
        fontsize=4.9,
        handletextpad=0.25,
        columnspacing=0.65,
        borderaxespad=0.3,
    )
    add_grid(ax_pareto)
    panel(ax_pareto, "b", x=-0.18, y=1.07)

    certified = [
        p
        for p in seed["policy"].unique()
        if p != "Zhu'26-style Aggregate-DPP"
    ]
    rank_data = d4_seed[d4_seed["policy"].isin(certified)].pivot(
        index="seed", columns="policy", values="wAoI"
    )
    ranks = rank_data.rank(axis=1, method="average", ascending=True)
    rank_order = ranks.median().sort_values().index.tolist()
    rank_long = (
        ranks.reset_index()
        .melt(id_vars="seed", var_name="policy", value_name="rank")
    )
    rank_long.to_csv(
        derived / "tmc_advanced_rank_stability.csv", index=False
    )
    ax_rank.axvspan(0.5, 3.5, color="#EDF6FB", zorder=0)
    rng = np.random.default_rng(20260728)
    for yy, policy in enumerate(rank_order):
        values = ranks[policy].to_numpy()
        color = COLORS[policy]
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        ax_rank.scatter(
            values,
            yy + jitter,
            s=6,
            color=color,
            alpha=0.30,
            linewidths=0,
            zorder=2,
        )
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        ax_rank.hlines(yy, q1, q3, color=color, linewidth=3.0, zorder=3)
        ax_rank.plot(
            med,
            yy,
            marker="|",
            markersize=7,
            color=BLACK,
            markeredgewidth=1.0,
            zorder=4,
        )
    ax_rank.set_yticks(
        np.arange(len(rank_order)), [SHORT[p] for p in rank_order]
    )
    ax_rank.invert_yaxis()
    ax_rank.set_xlim(0.5, len(rank_order) + 0.5)
    ax_rank.set_xticks(range(1, len(rank_order) + 1))
    ax_rank.set_xlabel("seed-level W-AoI rank at $d=4$ (lower better)")
    add_grid(ax_rank, "x")
    panel(ax_rank, "c", x=-0.18, y=1.07)

    save(fig, outdir / "fig_tmc_baseline_evidence")
    plt.close(fig)


def factorial_contrasts(factorial: pd.DataFrame) -> pd.DataFrame:
    frame = factorial.copy()
    frame["adaptive_belief"] = as_bool(frame["adaptive_belief"])
    frame["shield"] = as_bool(frame["shield"])
    frame["A"] = np.where(frame["adaptive_belief"], 1.0, -1.0)
    frame["E"] = np.where(
        frame["age_estimator"].eq("expected"), 1.0, -1.0
    )
    frame["S"] = np.where(frame["shield"], 1.0, -1.0)
    terms = {
        "Expected age": ("E",),
        "Bayesian belief": ("A",),
        "Safety shield": ("S",),
        "Belief × age": ("A", "E"),
        "Belief × shield": ("A", "S"),
        "Age × shield": ("E", "S"),
        "Three-way": ("A", "E", "S"),
    }
    rows = []
    for seed_id, part in frame.groupby("seed"):
        grand = float(part["wAoI"].mean())
        if len(part) != 8:
            continue
        for name, codes in terms.items():
            sign = np.ones(len(part))
            for code in codes:
                sign *= part[code].to_numpy()
            contrast = float(np.sum(sign * part["wAoI"]) / 4.0)
            rows.append(
                {
                    "seed": int(seed_id),
                    "effect": name,
                    "wAoI_effect": contrast,
                    "wAoI_effect_pct": 100.0 * contrast / grand,
                }
            )
    return pd.DataFrame(rows)


def factorial_figure(
    factorial: pd.DataFrame, derived: Path, outdir: Path
):
    frame = factorial.copy()
    frame["adaptive_belief"] = as_bool(frame["adaptive_belief"])
    frame["shield"] = as_bool(frame["shield"])
    contrasts = factorial_contrasts(frame)
    contrasts.to_csv(
        derived / "tmc_advanced_factorial_contrasts_seeds.csv", index=False
    )
    effect_summary = []
    for effect, part in contrasts.groupby("effect"):
        mean, half = mean_ci(part["wAoI_effect_pct"])
        effect_summary.append(
            {
                "effect": effect,
                "mean_effect_pct": mean,
                "ci95_halfwidth": half,
                "seeds": part["seed"].nunique(),
            }
        )
    effect_summary = pd.DataFrame(effect_summary)
    effect_summary.to_csv(
        derived / "tmc_advanced_factorial_contrasts_summary.csv",
        index=False,
    )

    fig = plt.figure(figsize=(183 * MM, 58 * MM))
    outer = fig.add_gridspec(
        1, 2, width_ratios=[1.18, 1.0], wspace=0.48
    )
    effect_grid = outer[0].subgridspec(
        1, 2, width_ratios=[0.36, 0.64], wspace=0.07
    )
    ax_effect_far = fig.add_subplot(effect_grid[0, 0])
    ax_effect_near = fig.add_subplot(
        effect_grid[0, 1], sharey=ax_effect_far
    )
    ax_trade = fig.add_subplot(outer[0, 1])

    order = [
        "Expected age",
        "Bayesian belief",
        "Safety shield",
        "Belief × age",
        "Belief × shield",
        "Age × shield",
        "Three-way",
    ]
    summary = effect_summary.set_index("effect").loc[order]
    y = np.arange(len(order))
    colors = [
        BLUE if name in {"Expected age", "Bayesian belief"} else DARK_GREY
        for name in order
    ]
    for yy, name, color in zip(y, order, colors):
        row = summary.loc[name]
        target = ax_effect_far if name == "Expected age" else ax_effect_near
        target.errorbar(
            row["mean_effect_pct"],
            yy,
            xerr=row["ci95_halfwidth"],
            fmt="o",
            markersize=3.5,
            color=color,
            capsize=1.8,
            elinewidth=0.85,
            zorder=3,
        )
    ax_effect_near.axvspan(-1.0, 1.0, color="#F3F3F3", zorder=0)
    ax_effect_near.axvline(
        0, color=BLACK, linestyle="--", linewidth=0.65
    )
    ax_effect_far.set_xlim(-39.0, -30.0)
    ax_effect_near.set_xlim(-1.5, 2.3)
    ax_effect_far.set_yticks(y, order)
    ax_effect_far.invert_yaxis()
    ax_effect_near.tick_params(axis="y", left=False, labelleft=False)
    ax_effect_far.spines["right"].set_visible(False)
    ax_effect_near.spines["left"].set_visible(False)
    ax_effect_far.set_xlabel("dominant effect (%)")
    ax_effect_near.set_xlabel("remaining effects (%)")
    add_grid(ax_effect_far, "x")
    add_grid(ax_effect_near, "x")
    break_size = 0.018
    break_kw = dict(color=BLACK, clip_on=False, linewidth=0.65)
    ax_effect_far.plot(
        (1 - break_size, 1 + break_size),
        (-break_size, +break_size),
        transform=ax_effect_far.transAxes,
        **break_kw,
    )
    ax_effect_far.plot(
        (1 - break_size, 1 + break_size),
        (1 - break_size, 1 + break_size),
        transform=ax_effect_far.transAxes,
        **break_kw,
    )
    ax_effect_near.plot(
        (-break_size, +break_size),
        (-break_size, +break_size),
        transform=ax_effect_near.transAxes,
        **break_kw,
    )
    ax_effect_near.plot(
        (-break_size, +break_size),
        (1 - break_size, 1 + break_size),
        transform=ax_effect_near.transAxes,
        **break_kw,
    )
    panel(ax_effect_far, "a", x=-0.55, y=1.08)

    trade_rows = []
    labels = {
        (False, "conservative"): "Static / cons.",
        (False, "expected"): "Static / exp.",
        (True, "conservative"): "Bayes / cons.",
        (True, "expected"): "Bayes / exp.",
    }
    color_map = {
        (False, "conservative"): MID_GREY,
        (False, "expected"): GREEN,
        (True, "conservative"): PURPLE,
        (True, "expected"): BLUE,
    }
    marker_map = {
        (False, "conservative"): "o",
        (False, "expected"): "s",
        (True, "conservative"): "D",
        (True, "expected"): "^",
    }
    offsets = {
        (False, "conservative"): (4, 4),
        (False, "expected"): (4, -12),
        (True, "conservative"): (4, 4),
        (True, "expected"): (-55, 5),
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
        aoi_cost = 100.0 * (
            aoi.loc[valid, True] - aoi.loc[valid, False]
        ) / aoi.loc[valid, False]
        power_reduction = 100.0 * (
            power.loc[valid, False] - power.loc[valid, True]
        ) / power.loc[valid, False]
        x, xci = mean_ci(aoi_cost)
        yval, yci = mean_ci(power_reduction)
        trade_rows.append(
            {
                "adaptive_belief": adaptive,
                "age_estimator": age_mode,
                "label": label,
                "aoi_cost_pct": x,
                "aoi_cost_ci95": xci,
                "power_reduction_pct": yval,
                "power_reduction_ci95": yci,
                "paired_seeds": len(valid),
            }
        )
        ax_trade.errorbar(
            x,
            yval,
            xerr=xci,
            yerr=yci,
            fmt=marker_map[key],
            markersize=4.0,
            color=color_map[key],
            capsize=1.8,
            elinewidth=0.8,
            label=label,
        )
    pd.DataFrame(trade_rows).to_csv(
        derived / "tmc_advanced_shield_tradeoff.csv", index=False
    )
    ax_trade.axhline(0, color=LIGHT_GREY, linewidth=0.6)
    ax_trade.axvline(0, color=LIGHT_GREY, linewidth=0.6)
    ax_trade.set_xlabel("W-AoI cost of shield (%)")
    ax_trade.set_ylabel("max-power reduction (%)")
    ax_trade.margins(x=0.10, y=0.18)
    ax_trade.legend(
        loc="upper right",
        fontsize=5.1,
        handlelength=1.1,
        labelspacing=0.2,
    )
    add_grid(ax_trade)
    panel(ax_trade, "b", x=-0.20, y=1.08)

    save(fig, outdir / "fig_tmc_factorial_effects")
    plt.close(fig)


def qcap_summary(qcap: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "wAoI",
        "max_avg_power",
        "worst_user_mean_aoi",
        "jain_service_fairness",
        "p95_slot_aoi",
        "starvation_p95",
        "rejection_rate",
        "p99_slot_aoi",
        "cvar95_slot_aoi",
    ]
    rows = []
    for cap, part in qcap.groupby("qcap"):
        row = {"qcap": float(cap), "seeds": part["seed"].nunique()}
        for metric in metrics:
            if metric not in part:
                continue
            mean, half = mean_ci(part[metric])
            row[metric] = mean
            row[metric + "_ci95"] = half
        rows.append(row)
    return pd.DataFrame(rows).sort_values("qcap")


def service_figure(qcap: pd.DataFrame, derived: Path, outdir: Path):
    summary = qcap_summary(qcap)
    summary.to_csv(
        derived / "tmc_advanced_qcap_summary.csv", index=False
    )
    caps = summary["qcap"].to_numpy()
    norm = mpl.colors.Normalize(vmin=float(caps.min()), vmax=float(caps.max()))
    cmap = mpl.colormaps["cividis"]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(183 * MM, 64 * MM),
        gridspec_kw={"wspace": 0.48},
    )
    specs = [
        (
            "max_avg_power",
            "wAoI",
            "maximum per-UAV power (mW)",
            "W-AoI (slots)",
        ),
        (
            "jain_service_fairness",
            "worst_user_mean_aoi",
            "Jain service fairness",
            "worst-user mean AoI (slots)",
        ),
        (
            "starvation_p95",
            "p95_slot_aoi",
            "p95 inter-success gap (slots)",
            "p95 slot-level AoI (slots)",
        ),
    ]
    label_offsets = {
        4.0: (3, 4),
        8.0: (3, -9),
        12.0: (3, 4),
        16.0: (3, -9),
        24.0: (-13, 5),
        28.0: (3, -9),
        32.0: (3, 5),
        36.0: (3, -9),
        48.0: (3, 4),
    }
    max_reject = max(float(summary["rejection_rate"].max()), 1e-9)
    for idx, (xmetric, ymetric, xlabel, ylabel) in enumerate(specs):
        ax = axes[idx]
        ax.plot(
            summary[xmetric],
            summary[ymetric],
            color="#A7A7A7",
            linewidth=0.8,
            zorder=1,
        )
        for _, row in summary.iterrows():
            cap = float(row["qcap"])
            size = 18 + 65 * float(row["rejection_rate"]) / max_reject
            ax.errorbar(
                row[xmetric],
                row[ymetric],
                xerr=row.get(xmetric + "_ci95", np.nan),
                yerr=row.get(ymetric + "_ci95", np.nan),
                fmt="none",
                ecolor=cmap(norm(cap)),
                elinewidth=0.65,
                capsize=1.4,
                alpha=0.75,
                zorder=2,
            )
            ax.scatter(
                row[xmetric],
                row[ymetric],
                s=size,
                color=cmap(norm(cap)),
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
            if cap in {4.0, 24.0, 48.0}:
                ax.annotate(
                    f"{cap:g}",
                    (row[xmetric], row[ymetric]),
                    xytext=label_offsets[cap],
                    textcoords="offset points",
                    fontsize=5.2,
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        add_grid(ax)
        panel(ax, chr(ord("a") + idx), x=-0.22, y=1.08)
    axes[0].axvline(0.4, color=BLACK, linestyle="--", linewidth=0.65)
    axes[0].text(
        0.399,
        0.98,
        "budget",
        transform=axes[0].get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=5.3,
    )
    axes[2].scatter(
        [], [], s=18, facecolor="none", edgecolor=DARK_GREY,
        linewidth=0.6, label="0%"
    )
    axes[2].scatter(
        [], [], s=83, facecolor="none", edgecolor=DARK_GREY,
        linewidth=0.6, label="33%"
    )
    axes[2].legend(
        title="candidate rejection",
        loc="upper right",
        fontsize=4.9,
        title_fontsize=4.9,
        handletextpad=0.3,
        labelspacing=0.25,
    )
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="horizontal",
        fraction=0.045,
        pad=0.17,
        aspect=45,
    )
    cbar.set_label(r"queue cap $Q_{\max}$", labelpad=1)
    cbar.ax.tick_params(labelsize=5.8, length=2)

    save(fig, outdir / "fig_tmc_service_pareto")
    plt.close(fig)


def robustness_figure(
    calibration: pd.DataFrame,
    shift: pd.DataFrame,
    measured: pd.DataFrame,
    derived: Path,
    outdir: Path,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(183 * MM, 63 * MM),
        gridspec_kw={"wspace": 0.48, "width_ratios": [0.90, 1.12, 1.12]},
    )
    ax_cal, ax_shift, ax_measured = axes

    cal = calibration.copy()
    dwell = sorted(cal["mean_dwell"].unique())
    delays = sorted(cal["extra_delay"].unique())
    gain = (
        cal.pivot(
            index="mean_dwell",
            columns="extra_delay",
            values="gain_vs_conservative_pct",
        )
        .reindex(index=dwell, columns=delays)
        .to_numpy()
    )
    gap = (
        cal.pivot(
            index="mean_dwell",
            columns="extra_delay",
            values="gap_vs_true_aoi_pct",
        )
        .reindex(index=dwell, columns=delays)
        .to_numpy()
    )
    image = ax_cal.imshow(
        gain,
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=max(42, float(np.nanmax(gain))),
    )
    for i in range(len(dwell)):
        for j in range(len(delays)):
            ax_cal.text(
                j,
                i,
                f"{gain[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="white" if gain[i, j] > 27 else BLACK,
            )
            if gap[i, j] > 5:
                ax_cal.plot(
                    j + 0.32,
                    i - 0.32,
                    marker="^",
                    markersize=2.6,
                    markerfacecolor=VERMILLION,
                    markeredgecolor=BLACK,
                    markeredgewidth=0.35,
                    color=BLACK,
                )
    ax_cal.set_xticks(range(len(delays)), [f"$d={int(x)}$" for x in delays])
    ax_cal.set_yticks(range(len(dwell)), [f"{x:g}" for x in dwell])
    ax_cal.set_xlabel("pending-feedback slots $d$")
    ax_cal.set_ylabel("mean NLoS dwell (slots)")
    ax_cal.text(
        0.5,
        1.03,
        "Expected-age gain (%)",
        transform=ax_cal.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.8,
    )
    for spine in ax_cal.spines.values():
        spine.set_visible(False)
    panel(ax_cal, "a", x=-0.25, y=1.13)
    cbar = fig.colorbar(image, ax=ax_cal, fraction=0.045, pad=0.03)
    cbar.ax.tick_params(labelsize=5.5, length=2)

    shift_order = [
        "FreshSky-Safe",
        "Wang'26 PORMAB-Safe",
        "Tripathi'24 Whittle-Safe",
        "Static-Geometry Safe",
        "Ji'24-style Greedy-Safe",
        "Zhu'26-style Aggregate-DPP",
    ]
    available = [p for p in shift_order if p in set(shift["policy"])]
    shift_rows = []
    x_positions = np.array([0.0, 1.0])
    for policy in available:
        part = shift[shift["policy"] == policy]
        pre, pre_ci = mean_ci(part["pre_wAoI"])
        post, post_ci = mean_ci(part["post_wAoI"])
        shift_rows.append(
            {
                "policy": policy,
                "pre_wAoI": pre,
                "pre_ci95": pre_ci,
                "post_wAoI": post,
                "post_ci95": post_ci,
                "seeds": part["seed"].nunique(),
            }
        )
        color = COLORS[policy]
        ax_shift.plot(
            x_positions,
            [pre, post],
            color=color,
            linewidth=1.0,
            alpha=0.9,
        )
        ax_shift.errorbar(
            x_positions,
            [pre, post],
            yerr=[pre_ci, post_ci],
            fmt=MARKERS[policy],
            color=color,
            markersize=3.7,
            capsize=1.7,
            elinewidth=0.75,
        )
        ax_shift.plot([], [], color=color, marker=MARKERS[policy],
                      linewidth=1.0, markersize=3.4,
                      label=SHORT[policy])
    pd.DataFrame(shift_rows).to_csv(
        derived / "tmc_advanced_shift_summary.csv", index=False
    )
    ax_shift.set_xlim(-0.18, 1.58)
    ax_shift.set_xticks([0, 1], ["pre-shift", "post-shift"])
    ax_shift.set_ylabel("W-AoI (slots)")
    ax_shift.legend(
        loc="lower center",
        bbox_to_anchor=(0.50, 1.01),
        ncol=2,
        fontsize=5.0,
        handlelength=1.5,
        columnspacing=0.7,
    )
    add_grid(ax_shift, "y")
    panel(ax_shift, "b", x=-0.20, y=1.08)

    expected_name = "Expected-AoI Bayes-Safe"
    conservative_name = "Conservative-AoI Bayes-Safe"
    measured_rows = []
    delay_styles = {4: (BLUE, "o"), 8: (ORANGE, "s")}
    for delay, (color, marker) in delay_styles.items():
        part = measured[measured["extra_delay"] == delay]
        means_x, means_y, ci_y = [], [], []
        for fraction, group in part.groupby("good_fraction"):
            pivot = group.pivot(
                index="seed", columns="policy", values="wAoI"
            )
            valid = pivot[[expected_name, conservative_name]].dropna()
            gain_values = 100.0 * (
                valid[conservative_name] - valid[expected_name]
            ) / valid[conservative_name]
            x = float(fraction)
            mean, half = mean_ci(gain_values)
            means_x.append(x)
            means_y.append(mean)
            ci_y.append(half)
            for seed_id, value in gain_values.items():
                measured_rows.append(
                    {
                        "good_fraction": x,
                        "extra_delay": int(delay),
                        "seed": int(seed_id),
                        "gain_vs_conservative_pct": float(value),
                    }
                )
            jitter = np.linspace(-0.010, 0.010, len(gain_values))
            ax_measured.scatter(
                x + jitter,
                gain_values,
                s=6,
                color=color,
                alpha=0.25,
                linewidths=0,
            )
        order_idx = np.argsort(means_x)
        means_x = np.asarray(means_x)[order_idx]
        means_y = np.asarray(means_y)[order_idx]
        ci_y = np.asarray(ci_y)[order_idx]
        ax_measured.errorbar(
            means_x,
            means_y,
            yerr=ci_y,
            color=color,
            marker=marker,
            linewidth=1.1,
            markersize=3.8,
            capsize=1.8,
            label=f"$d={delay}$",
            zorder=3,
        )
    pd.DataFrame(measured_rows).to_csv(
        derived / "tmc_advanced_measured_effects_seeds.csv", index=False
    )
    ax_measured.set_xlabel("measured good-state fraction")
    ax_measured.set_ylabel("expected-age gain (%)")
    ax_measured.legend(loc="upper left")
    add_grid(ax_measured)
    panel(ax_measured, "c", x=-0.20, y=1.08)

    save(fig, outdir / "fig_tmc_robustness")
    plt.close(fig)


def runtime_figure(runtime: pd.DataFrame, derived: Path, outdir: Path):
    runtime = runtime.copy()
    runtime["total_p95_ms"] = runtime["total_p95_us"] / 1000.0
    for field in (
        "expected_age_median_us",
        "decision_median_us",
        "accounting_median_us",
        "filter_update_median_us",
    ):
        runtime[field.replace("_us", "_ms")] = runtime[field] / 1000.0

    focus = runtime[
        (runtime["retained_models"] == 144)
        & (runtime["extra_delay"] == 8)
    ].sort_values("N")
    focus.to_csv(
        derived / "tmc_advanced_runtime_focus.csv", index=False
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(88 * MM, 86 * MM),
        gridspec_kw={"height_ratios": [1.02, 1.0], "hspace": 0.62},
    )
    ax_stack, ax_heat = axes
    y = np.arange(len(focus))
    left = np.zeros(len(focus))
    components = [
        ("expected_age_median_ms", "receiver-age update", SKY),
        ("decision_median_ms", "ranking + shield", BLUE),
        ("accounting_median_ms", "state accounting", GREEN),
        ("filter_update_median_ms", "Bayesian filter", ORANGE),
    ]
    for field, label, color in components:
        values = focus[field].to_numpy()
        ax_stack.barh(
            y,
            values,
            left=left,
            height=0.58,
            color=color,
            label=label,
            edgecolor="white",
            linewidth=0.25,
        )
        left += values
    ax_stack.scatter(
        focus["total_p95_ms"],
        y,
        color=BLACK,
        marker="D",
        s=15,
        label="end-to-end p95",
        zorder=4,
    )
    ax_stack.set_yticks(y, [f"$N={int(n)}$" for n in focus["N"]])
    ax_stack.invert_yaxis()
    ax_stack.set_xlabel("decision time (ms), $K=144$, $d=8$")
    ax_stack.legend(
        loc="upper right",
        ncol=2,
        handlelength=1.2,
        labelspacing=0.2,
    )
    add_grid(ax_stack, "x")
    panel(ax_stack, "a", x=-0.25, y=1.08)

    heat = runtime[runtime["extra_delay"] == 8].pivot(
        index="N", columns="retained_models", values="total_p95_ms"
    )
    heat = heat.sort_index().sort_index(axis=1)
    values = heat.to_numpy()
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
            rgba = image.cmap(image.norm(values[i, j]))
            luminance = (
                0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            )
            ax_heat.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if luminance < 0.48 else BLACK,
            )
    ax_heat.set_xticks(
        range(len(heat.columns)), [f"{int(k)}" for k in heat.columns]
    )
    ax_heat.set_yticks(
        range(len(heat.index)), [f"{int(n)}" for n in heat.index]
    )
    ax_heat.set_xlabel("retained channel models $K$")
    ax_heat.set_ylabel("number of UAVs $N$")
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.035, pad=0.03)
    cbar.set_label("p95 time (ms)", labelpad=2)
    cbar.ax.tick_params(labelsize=5.5, length=2)
    panel(ax_heat, "b", x=-0.25, y=1.08)

    save(fig, outdir / "fig_tmc_runtime")
    plt.close(fig)


def export_uav_long(seed: pd.DataFrame, derived: Path):
    vector_fields = {
        "uav_pi_los": "pi_los",
        "uav_pth": "pth",
        "uav_mean_aoi": "mean_aoi",
        "uav_avg_power": "avg_power",
        "uav_service_rate": "service_rate",
        "uav_max_queue": "max_queue",
    }
    if not set(vector_fields).issubset(seed.columns):
        return
    rows = []
    for _, row in seed.iterrows():
        vectors = {
            target: np.fromstring(str(row[source]), sep=";")
            for source, target in vector_fields.items()
        }
        lengths = {len(values) for values in vectors.values()}
        if len(lengths) != 1:
            raise ValueError("inconsistent serialized per-UAV vector lengths")
        for uav in range(next(iter(lengths))):
            item = {
                "policy": row["policy"],
                "extra_delay": int(row["extra_delay"]),
                "seed": int(row["seed"]),
                "uav": uav,
            }
            item.update({name: values[uav] for name, values in vectors.items()})
            rows.append(item)
    pd.DataFrame(rows).to_csv(
        derived / "tmc_extended_uav_long.csv", index=False
    )


def validate(
    baseline: pd.DataFrame,
    factorial: pd.DataFrame,
    qcap: pd.DataFrame,
):
    expected = set(SHORT)
    found = set(baseline["policy"].unique())
    if expected != found:
        raise ValueError(
            f"baseline policy set mismatch: missing={expected-found}, "
            f"extra={found-expected}"
        )
    if set(baseline["extra_delay"]) != {2, 4, 8}:
        raise ValueError("baseline delays must be 2, 4, and 8")
    if baseline.groupby(["policy", "extra_delay"])["seed"].nunique().min() != 20:
        raise ValueError("every baseline cell must have 20 seeds")
    if factorial.groupby(
        ["adaptive_belief", "age_estimator", "shield"]
    )["seed"].nunique().min() != 20:
        raise ValueError("every factorial cell must have 20 seeds")
    expected_caps = {4, 8, 12, 16, 24, 28, 32, 36, 48}
    if set(qcap["qcap"]) != expected_caps:
        raise ValueError("Qmax grid mismatch")
    if qcap.groupby("qcap")["seed"].nunique().min() != 20:
        raise ValueError("every Qmax point must have 20 seeds")
    for frame in (baseline, factorial, qcap):
        numeric = frame.select_dtypes(include="number")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("non-finite numeric result")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--extended-results", required=True, type=Path)
    parser.add_argument("--legacy-results", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--derived-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.derived_dir.mkdir(parents=True, exist_ok=True)

    source = args.extended_results
    legacy = (
        args.legacy_results
        if args.legacy_results is not None
        else args.project_root / "experiments" / "results"
    )
    baseline = pd.read_csv(source / "tmc_extended_baselines_seeds.csv")
    factorial = pd.read_csv(source / "tmc_factorial_ablation_seeds.csv")
    qcap = pd.read_csv(source / "tmc_qcap_service_seeds.csv")
    shift = pd.read_csv(source / "tmc_delayed_shift_seeds.csv")
    calibration = pd.read_csv(legacy / "tmc_aoi_calibration_seeds.csv")
    measured = pd.read_csv(legacy / "measured_a2g_delayed_aoi_seeds.csv")
    runtime = pd.read_csv(legacy / "tmc_runtime_scaling.csv")

    validate(baseline, factorial, qcap)
    export_uav_long(baseline, args.derived_dir)
    baseline_figure(baseline, args.derived_dir, args.output_dir)
    baseline_figure_v3(source, args.output_dir, args.derived_dir)
    factorial_figure_v3(factorial, args.derived_dir, args.output_dir)
    service_figure_v3(qcap, args.output_dir, args.derived_dir)
    robustness_figure_v3(
        calibration,
        measured,
        args.derived_dir,
        args.output_dir,
    )
    runtime_figure_v3(runtime, args.derived_dir, args.output_dir)
    print(
        "generated five journal-only advanced figures and derived CSVs"
    )


if __name__ == "__main__":
    main()
