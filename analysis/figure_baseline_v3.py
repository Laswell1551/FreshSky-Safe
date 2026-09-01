# -*- coding: utf-8 -*-
"""Fig. 4 v3 prototype for the FreshSky-Safe TMC manuscript.

This standalone prototype reads only executed CSV results.  It does not
modify the manuscript, the main plotting script, or any experiment output.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle


MM = 1.0 / 25.4

# Okabe-Ito plus neutral support colors.
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
BLACK = "#222222"
DARK_GREY = "#666666"
MID_GREY = "#9B9B9B"
LIGHT_GREY = "#D9D9D9"
PALE_GREY = "#F2F2F2"

POLICY_ORDER = [
    "FreshSky-Safe",
    "Conservative-AoI Safe",
    "Max-Age-First Safe",
    "Round-Robin Safe",
    "Ji'24-style Greedy-Safe",
    "Zhao'25 MMSE-MW-Safe",
    "Static-Geometry Safe",
    "Known-Transition Safe",
    "Tripathi'24 Whittle-Safe",
    "Wang'26 PORMAB-Safe",
    "Zhu'26-style Aggregate-DPP",
]

SHORT = {
    "FreshSky-Safe": "FreshSky",
    "Conservative-AoI Safe": "Conservative age",
    "Max-Age-First Safe": "Max-Age",
    "Round-Robin Safe": "Round-Robin",
    "Ji'24-style Greedy-Safe": "Ji'24 greedy",
    "Zhao'25 MMSE-MW-Safe": "Zhao'25 MW-E",
    "Static-Geometry Safe": "Static DPP-MC",
    "Known-Transition Safe": "Known transition",
    "Tripathi'24 Whittle-Safe": "Tripathi'24 WI",
    "Wang'26 PORMAB-Safe": "Wang'26 PORMAB",
    "Zhu'26-style Aggregate-DPP": "Zhu'26 agg.-DPP",
}

COLORS = {
    "FreshSky-Safe": BLUE,
    "Conservative-AoI Safe": MID_GREY,
    "Max-Age-First Safe": MID_GREY,
    "Round-Robin Safe": DARK_GREY,
    "Ji'24-style Greedy-Safe": SKY,
    "Zhao'25 MMSE-MW-Safe": "#7A5195",
    "Static-Geometry Safe": GREEN,
    "Known-Transition Safe": PURPLE,
    "Tripathi'24 Whittle-Safe": "#B66D00",
    "Wang'26 PORMAB-Safe": ORANGE,
    "Zhu'26-style Aggregate-DPP": VERMILLION,
}

MARKERS = {
    "FreshSky-Safe": "o",
    "Conservative-AoI Safe": "<",
    "Max-Age-First Safe": "h",
    "Round-Robin Safe": ">",
    "Ji'24-style Greedy-Safe": "s",
    "Zhao'25 MMSE-MW-Safe": "p",
    "Static-Geometry Safe": "P",
    "Known-Transition Safe": "D",
    "Tripathi'24 Whittle-Safe": "v",
    "Wang'26 PORMAB-Safe": "^",
    "Zhu'26-style Aggregate-DPP": "X",
}

T_CRITICAL_20 = 2.093024054408263

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
        "font.size": 6.4,
        "axes.titlesize": 6.8,
        "axes.labelsize": 6.4,
        "xtick.labelsize": 5.6,
        "ytick.labelsize": 5.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 450,
    }
)


def mean_ci(values: pd.Series | np.ndarray) -> tuple[float, float]:
    """Mean and two-sided 95% t interval half-width."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, float("nan")
    tcrit = T_CRITICAL_20 if len(values) == 20 else 1.96
    half = tcrit * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return mean, half


def fmt_signed(value: float, digits: int = 1) -> str:
    text = f"{value:+.{digits}f}"
    return text.replace("-", "\N{MINUS SIGN}")


def validate_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def load_data(data_root: Path):
    effects_path = data_root / "derived" / "tmc_advanced_baseline_effects.csv"
    pareto_path = data_root / "derived" / "tmc_advanced_baseline_pareto.csv"
    rank_path = data_root / "derived" / "tmc_advanced_rank_stability.csv"
    shift_path = data_root / "tmc_delayed_shift_seeds.csv"

    for path in (effects_path, pareto_path, rank_path, shift_path):
        if not path.exists():
            raise FileNotFoundError(path)

    effects = pd.read_csv(effects_path)
    pareto = pd.read_csv(pareto_path)
    ranks = pd.read_csv(rank_path)
    shift = pd.read_csv(shift_path)

    validate_columns(
        effects,
        {
            "extra_delay",
            "baseline",
            "freshsky_gain_pct",
            "ci95_halfwidth",
            "paired_seeds",
        },
        effects_path,
    )
    validate_columns(
        pareto,
        {
            "policy",
            "max_avg_power",
            "max_avg_power_ci95",
            "wAoI",
            "wAoI_ci95",
        },
        pareto_path,
    )
    validate_columns(ranks, {"seed", "policy", "rank"}, rank_path)
    validate_columns(
        shift,
        {"policy", "seed", "pre_wAoI", "post_wAoI", "certificate_ok"},
        shift_path,
    )
    return effects, pareto, ranks, shift


def shift_differences(shift: pd.DataFrame) -> pd.DataFrame:
    """Paired FreshSky-minus-comparator pre/post W-AoI differences."""
    ours_name = "FreshSky-Safe"
    comparators = [
        "Ji'24-style Greedy-Safe",
        "Static-Geometry Safe",
        "Zhu'26-style Aggregate-DPP",
    ]
    ours = shift[shift["policy"].eq(ours_name)].set_index("seed")
    rows: list[dict[str, float | int | str | bool]] = []
    for comparator in comparators:
        other = shift[shift["policy"].eq(comparator)].set_index("seed")
        seeds = ours.index.intersection(other.index)
        for phase, column in (("pre", "pre_wAoI"), ("post", "post_wAoI")):
            delta = ours.loc[seeds, column] - other.loc[seeds, column]
            mean, half = mean_ci(delta)
            rows.append(
                {
                    "comparator": comparator,
                    "phase": phase,
                    "mean_delta": mean,
                    "ci95_halfwidth": half,
                    "paired_seeds": len(delta),
                    "significant": bool((mean - half > 0) or (mean + half < 0)),
                }
            )
    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(
        True,
        axis=grid_axis,
        color="#E5E5E5",
        linewidth=0.5,
        zorder=0,
    )


def add_panel_heading(
    ax: plt.Axes,
    letter: str,
    title: str,
    x: float = -0.02,
    y: float = 1.14,
) -> None:
    ax.text(
        x,
        y,
        f"({letter})  {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        fontweight="bold",
    )


def draw_effect_matrix(
    ax: plt.Axes,
    ax_rank: plt.Axes,
    effects: pd.DataFrame,
    ranks: pd.DataFrame,
) -> None:
    delay_values = [2, 4, 8]
    effect_lookup = effects.set_index(["baseline", "extra_delay"])
    y_lookup = {policy: i for i, policy in enumerate(POLICY_ORDER)}
    n_rows = len(POLICY_ORDER)

    effect_cmap = LinearSegmentedColormap.from_list(
        "freshsky_effect",
        ["#F5B79C", "#FFFFFF", "#98C9E3"],
    )
    effect_norm = TwoSlopeNorm(vmin=-10.5, vcenter=0.0, vmax=42.5)

    for i, policy in enumerate(POLICY_ORDER):
        for j, delay in enumerate(delay_values):
            if policy == "FreshSky-Safe":
                face = "#E5F2F8"
                mean = 0.0
                half = 0.0
                label = "REF"
                ci_text = "paired anchor"
                significant = False
            else:
                row = effect_lookup.loc[(policy, delay)]
                mean = float(row["freshsky_gain_pct"])
                half = float(row["ci95_halfwidth"])
                face = effect_cmap(effect_norm(mean))
                label = fmt_signed(mean)
                ci_text = (
                    f"[{fmt_signed(mean - half)}, {fmt_signed(mean + half)}]"
                )
                significant = (mean - half > 0) or (mean + half < 0)

            ax.add_patch(
                Rectangle(
                    (j - 0.47, i - 0.42),
                    0.94,
                    0.84,
                    facecolor=face,
                    edgecolor="white",
                    linewidth=0.75,
                    zorder=1,
                )
            )
            ax.text(
                j,
                i - 0.075,
                label,
                ha="center",
                va="center",
                fontsize=5.45,
                color=BLACK,
                fontweight="bold" if (significant or policy == "FreshSky-Safe") else "normal",
                zorder=3,
            )
            ax.text(
                j,
                i + 0.19,
                ci_text,
                ha="center",
                va="center",
                fontsize=4.05,
                color=DARK_GREY,
                zorder=3,
            )
            if policy != "FreshSky-Safe":
                ax.scatter(
                    [j + 0.355],
                    [i - 0.255],
                    s=5.8,
                    marker="o",
                    facecolors=BLACK if significant else "white",
                    edgecolors=BLACK,
                    linewidths=0.45,
                    zorder=4,
                )

        ax.scatter(
            [-0.60],
            [i],
            s=14,
            marker=MARKERS[policy],
            facecolors=(
                "white" if policy == "Known-Transition Safe" else COLORS[policy]
            ),
            edgecolors=COLORS[policy],
            linewidths=0.8,
            clip_on=False,
            zorder=5,
        )

    separators = [0.5, 3.5, 7.5, 9.5]
    for y in separators:
        ax.axhline(y, color=LIGHT_GREY, linewidth=0.75, zorder=5)
        ax_rank.axhline(y, color=LIGHT_GREY, linewidth=0.75, zorder=1)

    ax.set_xlim(-0.72, 2.49)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(range(3), [f"$d={d}$" for d in delay_values])
    ax.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        pad=1.5,
    )
    ax.set_yticks(
        range(n_rows),
        [SHORT[policy] for policy in POLICY_ORDER],
    )
    ax.tick_params(axis="y", length=0, pad=4.5)
    for label, policy in zip(ax.get_yticklabels(), POLICY_ORDER):
        label.set_color(
            BLUE
            if policy == "FreshSky-Safe"
            else (
                VERMILLION
                if policy == "Zhu'26-style Aggregate-DPP"
                else BLACK
            )
        )
        if policy in {"FreshSky-Safe", "Zhu'26-style Aggregate-DPP"}:
            label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)

    add_panel_heading(ax, "a", "Paired improvement matrix + rank stability", x=-0.42)
    ax.text(
        0.0,
        1.07,
        "cell = mean [95% CI];  + FreshSky better;  "
        r"$\bullet$ CI excludes 0",
        transform=ax.transAxes,
        fontsize=4.35,
        color=DARK_GREY,
        ha="left",
        va="bottom",
    )

    # Compact d=4 rank strip, aligned with the same policy rows.
    ax_rank.axvspan(0.5, 3.5, color="#EDF6FB", zorder=0)
    rank_groups = {p: g["rank"].to_numpy() for p, g in ranks.groupby("policy")}
    for policy in POLICY_ORDER:
        i = y_lookup[policy]
        if policy not in rank_groups:
            ax_rank.text(
                5.0,
                i,
                "n/a\naggregate",
                ha="center",
                va="center",
                fontsize=4.2,
                color=VERMILLION,
                linespacing=0.85,
            )
            continue
        values = np.asarray(rank_groups[policy], dtype=float)
        q0, q1, q2, q3, q4 = np.quantile(values, [0, 0.25, 0.5, 0.75, 1])
        color = COLORS[policy]
        ax_rank.hlines(i, q0, q4, color=LIGHT_GREY, linewidth=0.8, zorder=2)
        ax_rank.hlines(i, q1, q3, color=color, linewidth=3.2, zorder=3)
        ax_rank.vlines(q2, i - 0.20, i + 0.20, color=BLACK, linewidth=0.75, zorder=4)
        ax_rank.text(
            min(q4 + 0.18, 10.25),
            i,
            f"{q2:g}",
            fontsize=4.2,
            color=DARK_GREY,
            ha="left",
            va="center",
        )
    ax_rank.set_xlim(0.5, 10.55)
    ax_rank.set_ylim(n_rows - 0.5, -0.5)
    ax_rank.set_xticks([1, 5, 10])
    ax_rank.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        length=2.0,
        pad=1.5,
    )
    ax_rank.tick_params(axis="y", left=False, labelleft=False)
    ax_rank.set_title("rank @ $d=4$\n1 = best", fontsize=5.5, pad=2.0)
    ax_rank.grid(True, axis="x", color="#E5E5E5", linewidth=0.45, zorder=0)
    for spine in ax_rank.spines.values():
        spine.set_visible(False)


def draw_contract_plane(
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    pareto: pd.DataFrame,
) -> None:
    left_policies = [
        "FreshSky-Safe",
        "Ji'24-style Greedy-Safe",
        "Zhao'25 MMSE-MW-Safe",
        "Static-Geometry Safe",
        "Known-Transition Safe",
        "Tripathi'24 Whittle-Safe",
        "Wang'26 PORMAB-Safe",
    ]
    zhu_name = "Zhu'26-style Aggregate-DPP"

    ax_left.axvspan(0.388, 0.4, color="#E7F4EE", zorder=0)
    ax_left.axvline(0.4, color=BLACK, linestyle="--", linewidth=0.75, zorder=1)
    ax_left.text(
        0.39975,
        487.0,
        "individual\nbudget",
        ha="right",
        va="top",
        fontsize=4.5,
        color=DARK_GREY,
    )

    for policy in left_policies:
        row = pareto[pareto["policy"].eq(policy)]
        if row.empty:
            continue
        row = row.iloc[0]
        color = COLORS[policy]
        hollow = policy == "Known-Transition Safe"
        ax_left.errorbar(
            float(row["max_avg_power"]),
            float(row["wAoI"]),
            xerr=float(row["max_avg_power_ci95"]),
            yerr=float(row["wAoI_ci95"]),
            fmt=MARKERS[policy],
            markersize=4.2,
            markerfacecolor="white" if hollow else color,
            markeredgecolor=color,
            markeredgewidth=0.9,
            color=color,
            ecolor=mpl.colors.to_rgba(color, 0.60),
            elinewidth=0.65,
            capsize=1.6,
            zorder=4,
        )

    label_targets = {
        "Ji'24-style Greedy-Safe": (0.4022, 480.0),
        "FreshSky-Safe": (0.4022, 469.0),
        "Known-Transition Safe": (0.4022, 458.0),
        "Static-Geometry Safe": (0.4022, 447.0),
        "Zhao'25 MMSE-MW-Safe": (0.4022, 436.0),
        "Wang'26 PORMAB-Safe": (0.4022, 425.0),
        "Tripathi'24 Whittle-Safe": (0.4022, 414.0),
    }
    direct = {
        "FreshSky-Safe": "FreshSky",
        "Ji'24-style Greedy-Safe": "Ji'24",
        "Zhao'25 MMSE-MW-Safe": "Zhao'25",
        "Known-Transition Safe": "Known (info+)",
        "Static-Geometry Safe": "Static",
        "Wang'26 PORMAB-Safe": "Wang'26",
        "Tripathi'24 Whittle-Safe": "Tripathi'24",
    }
    for policy, xytext in label_targets.items():
        row = pareto[pareto["policy"].eq(policy)].iloc[0]
        ax_left.annotate(
            direct[policy],
            xy=(float(row["max_avg_power"]), float(row["wAoI"])),
            xytext=xytext,
            textcoords="data",
            fontsize=4.35,
            color=COLORS[policy],
            fontweight="bold" if policy == "FreshSky-Safe" else "normal",
            ha="left",
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": mpl.colors.to_rgba(COLORS[policy], 0.75),
                "linewidth": 0.5,
                "shrinkA": 1,
                "shrinkB": 2,
            },
            zorder=6,
        )

    zhu = pareto[pareto["policy"].eq(zhu_name)].iloc[0]
    ax_right.errorbar(
        float(zhu["max_avg_power"]),
        float(zhu["wAoI"]),
        xerr=float(zhu["max_avg_power_ci95"]),
        yerr=float(zhu["wAoI_ci95"]),
        fmt=MARKERS[zhu_name],
        markersize=5.2,
        markerfacecolor=VERMILLION,
        markeredgecolor=VERMILLION,
        color=VERMILLION,
        ecolor=mpl.colors.to_rgba(VERMILLION, 0.68),
        elinewidth=0.7,
        capsize=1.6,
        zorder=4,
    )
    ax_right.annotate(
        "Zhu'26\naggregate\nuncertified",
        xy=(float(zhu["max_avg_power"]), float(zhu["wAoI"])),
        xytext=(0.589, 452),
        fontsize=4.35,
        color=VERMILLION,
        ha="right",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": VERMILLION,
            "linewidth": 0.55,
        },
    )

    for ax in (ax_left, ax_right):
        ax.set_ylim(375, 492)
        style_axis(ax)
    ax_left.set_xlim(0.3885, 0.4060)
    ax_right.set_xlim(0.523, 0.597)
    ax_left.set_xticks([0.39, 0.40])
    ax_right.set_xticks([0.56])
    ax_left.set_ylabel("W-AoI (slots)")
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)

    # Broken-axis marks make the contract discontinuity explicit.
    break_kw = dict(color=BLACK, clip_on=False, linewidth=0.6)
    d = 0.018
    ax_left.plot((1 - d, 1 + d), (-d, +d), transform=ax_left.transAxes, **break_kw)
    ax_left.plot(
        (1 - d, 1 + d),
        (1 - d, 1 + d),
        transform=ax_left.transAxes,
        **break_kw,
    )
    ax_right.plot((-d, +d), (-d, +d), transform=ax_right.transAxes, **break_kw)
    ax_right.plot(
        (-d, +d),
        (1 - d, 1 + d),
        transform=ax_right.transAxes,
        **break_kw,
    )

    add_panel_heading(ax_left, "b", "$d=4$ contract plane", x=-0.24)
    ax_left.text(
        0.95,
        -0.27,
        "maximum per-UAV power (mW)",
        transform=ax_left.transAxes,
        ha="center",
        va="top",
        fontsize=6.2,
    )


def draw_shift_forest(ax: plt.Axes, shift_summary: pd.DataFrame) -> None:
    order = [
        "Ji'24-style Greedy-Safe",
        "Static-Geometry Safe",
        "Zhu'26-style Aggregate-DPP",
    ]
    short = {
        "Ji'24-style Greedy-Safe": "Ji'24 greedy",
        "Static-Geometry Safe": "Static DPP-MC",
        "Zhu'26-style Aggregate-DPP": "Zhu'26 agg.-DPP",
    }
    y_lookup = {policy: i for i, policy in enumerate(order)}
    low = float(
        np.min(
            shift_summary["mean_delta"]
            - shift_summary["ci95_halfwidth"]
        )
    )
    high = float(
        np.max(
            shift_summary["mean_delta"]
            + shift_summary["ci95_halfwidth"]
        )
    )
    xmin = min(-16.0, math.floor(low / 5.0) * 5.0 - 2.0)
    xmax = max(58.0, math.ceil(high / 5.0) * 5.0 + 5.0)
    ax.axvspan(xmin, 0, color="#EDF6FB", zorder=0)
    ax.axvspan(0, xmax, color="#FFF3EC", zorder=0)
    ax.axvline(0, color=BLACK, linestyle="--", linewidth=0.7, zorder=1)

    for policy in order:
        base_y = y_lookup[policy]
        color = COLORS[policy]
        for phase, offset, marker in (("pre", -0.14, "o"), ("post", 0.14, "s")):
            row = shift_summary[
                shift_summary["comparator"].eq(policy)
                & shift_summary["phase"].eq(phase)
            ].iloc[0]
            mean = float(row["mean_delta"])
            half = float(row["ci95_halfwidth"])
            y = base_y + offset
            face = "white" if phase == "pre" else color
            ax.errorbar(
                mean,
                y,
                xerr=half,
                fmt=marker,
                markersize=4.0,
                markerfacecolor=face,
                markeredgecolor=color,
                markeredgewidth=0.85,
                color=color,
                ecolor=mpl.colors.to_rgba(color, 0.72),
                elinewidth=0.75,
                capsize=1.6,
                zorder=4,
            )
            text_x = mean + half + 1.2
            ax.text(
                text_x,
                y,
                fmt_signed(mean),
                fontsize=4.3,
                color=color,
                ha="left",
                va="center",
                fontweight="bold" if bool(row["significant"]) else "normal",
            )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(len(order) - 0.55, -0.55)
    ax.set_yticks(range(len(order)), [short[p] for p in order])
    ax.tick_params(axis="y", length=0, pad=3.5)
    for label, policy in zip(ax.get_yticklabels(), order):
        label.set_color(COLORS[policy])
        if policy == "Zhu'26-style Aggregate-DPP":
            label.set_fontweight("bold")
    ax.set_xlabel(
        r"paired $\Delta$ W-AoI = FreshSky $-$ comparator (slots)"
    )
    style_axis(ax, "x")
    ax.text(
        0.01,
        0.98,
        "FreshSky better",
        transform=ax.transAxes,
        fontsize=4.5,
        color=BLUE,
        ha="left",
        va="top",
    )
    ax.text(
        0.99,
        0.98,
        "FreshSky worse",
        transform=ax.transAxes,
        fontsize=4.5,
        color=VERMILLION,
        ha="right",
        va="top",
    )
    ax.scatter(
        [0.61],
        [1.10],
        transform=ax.transAxes,
        s=11,
        marker="o",
        facecolors="white",
        edgecolors=DARK_GREY,
        linewidths=0.7,
        clip_on=False,
    )
    ax.text(
        0.635,
        1.10,
        "pre",
        transform=ax.transAxes,
        fontsize=4.5,
        ha="left",
        va="center",
    )
    ax.scatter(
        [0.77],
        [1.10],
        transform=ax.transAxes,
        s=11,
        marker="s",
        facecolors=DARK_GREY,
        edgecolors=DARK_GREY,
        linewidths=0.7,
        clip_on=False,
    )
    ax.text(
        0.795,
        1.10,
        "post",
        transform=ax.transAxes,
        fontsize=4.5,
        ha="left",
        va="center",
    )
    add_panel_heading(ax, "c", "Shift boundary: paired pre/post", x=-0.20)


def build_figure(
    effects: pd.DataFrame,
    pareto: pd.DataFrame,
    ranks: pd.DataFrame,
    shift: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame]:
    fig = plt.figure(figsize=(183 * MM, 98 * MM), facecolor="white")
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.70, 1.0],
        height_ratios=[1.08, 0.92],
        left=0.165,
        right=0.985,
        top=0.85,
        bottom=0.105,
        wspace=0.34,
        hspace=0.62,
    )
    matrix_grid = outer[:, 0].subgridspec(
        1, 2, width_ratios=[3.45, 1.0], wspace=0.09
    )
    ax_matrix = fig.add_subplot(matrix_grid[0, 0])
    ax_rank = fig.add_subplot(matrix_grid[0, 1], sharey=ax_matrix)
    draw_effect_matrix(ax_matrix, ax_rank, effects, ranks)

    contract_grid = outer[0, 1].subgridspec(
        1, 2, width_ratios=[0.78, 0.22], wspace=0.08
    )
    ax_contract = fig.add_subplot(contract_grid[0, 0])
    ax_zhu = fig.add_subplot(contract_grid[0, 1], sharey=ax_contract)
    draw_contract_plane(ax_contract, ax_zhu, pareto)

    shift_summary = shift_differences(shift)
    ax_shift = fig.add_subplot(outer[1, 1])
    draw_shift_forest(ax_shift, shift_summary)

    return fig, shift_summary


def save_outputs(
    fig: plt.Figure,
    outdir: Path,
    stem: str = "fig_tmc_baseline_evidence",
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    base = outdir / stem
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(base.with_suffix(".svg"), facecolor="white")
    fig.savefig(base.with_suffix(".png"), dpi=450, facecolor="white")


def make_figure(
    data_root: Path,
    outdir: Path,
    derived: Path | None = None,
    stem: str = "fig_tmc_baseline_evidence",
) -> None:
    """Build Fig. 4 and export both shift views used by the manuscript."""
    derived = data_root / "derived" if derived is None else derived
    derived.mkdir(parents=True, exist_ok=True)
    effects, pareto, ranks, shift = load_data(data_root)
    fig, shift_summary = build_figure(effects, pareto, ranks, shift)
    save_outputs(fig, outdir, stem=stem)
    plt.close(fig)
    shift_summary.to_csv(
        derived / "tmc_v3_shift_differences.csv", index=False
    )

    legacy_rows = []
    for policy, part in shift.groupby("policy"):
        pre, pre_ci = mean_ci(part["pre_wAoI"])
        post, post_ci = mean_ci(part["post_wAoI"])
        legacy_rows.append(
            {
                "policy": policy,
                "pre_wAoI": pre,
                "pre_ci95": pre_ci,
                "post_wAoI": post,
                "post_ci95": post_ci,
                "seeds": part["seed"].nunique(),
            }
        )
    pd.DataFrame(legacy_rows).to_csv(
        derived / "tmc_advanced_shift_summary.csv", index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "results"
            / "tmc_extended"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "out",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    effects, pareto, ranks, shift = load_data(args.data_root)
    fig, shift_summary = build_figure(effects, pareto, ranks, shift)
    save_outputs(fig, args.outdir, stem="fig4_v3")
    plt.close(fig)

    print("Saved:")
    for suffix in (".pdf", ".svg", ".png"):
        print(args.outdir / f"fig4_v3{suffix}")
    print("\nShift boundary (FreshSky - comparator, slots):")
    print(
        shift_summary[
            [
                "comparator",
                "phase",
                "mean_delta",
                "ci95_halfwidth",
                "paired_seeds",
                "significant",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
