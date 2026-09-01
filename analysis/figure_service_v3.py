#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fig. 6 v3 prototype from strict/v2 Qmax seed outcomes.

The script is intentionally standalone and does not modify the manuscript or
the existing figure-generation pipeline.  All uncertainty is computed across
the 20 random seeds.  Serialized per-UAV vectors are first reduced within each
seed, preserving the seed as the inferential unit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from scipy.stats import t


MM = 1.0 / 25.4
EXPECTED_CAPS = [4.0, 8.0, 12.0, 16.0, 24.0, 28.0, 32.0, 36.0, 48.0]
PBAR_MW = 0.4

BLACK = "#202124"
DARK_GREY = "#5B6168"
LIGHT_GREY = "#D9DDE1"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#7A5195"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 6.2,
            "axes.labelsize": 6.5,
            "axes.titlesize": 6.8,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 5.1,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
        }
    )


def mean_ci(values: pd.Series | np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float(np.mean(x)), float("nan")
    half = float(t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x)))
    return float(np.mean(x)), half


def parse_vector(value: object, field: str) -> np.ndarray:
    vector = np.fromstring(str(value), sep=";")
    if len(vector) != 12 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{field}: expected 12 finite UAV values, got {len(vector)}")
    return vector


def validate(frame: pd.DataFrame) -> None:
    required = {
        "policy",
        "extra_delay",
        "seed",
        "qcap",
        "wAoI",
        "worst_user_mean_aoi",
        "p95_user_mean_aoi",
        "p95_slot_aoi",
        "p99_slot_aoi",
        "cvar95_slot_aoi",
        "slot_aoi_gt40",
        "slot_aoi_gt60",
        "worst_service_rate",
        "jain_service_fairness",
        "starvation_p95",
        "starvation_max",
        "rejection_rate",
        "max_avg_power",
        "finite_window_bound",
        "certificate_ok",
        "uav_avg_power",
        "uav_max_queue",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing fields: {missing}")
    caps = sorted(frame["qcap"].astype(float).unique().tolist())
    if caps != EXPECTED_CAPS:
        raise ValueError(f"Qmax grid mismatch: {caps}")
    counts = frame.groupby("qcap")["seed"].nunique()
    if not (counts == 20).all() or len(frame) != 180:
        raise ValueError(f"expected 9 x 20 seed rows, got {len(frame)}; {counts.to_dict()}")
    if frame.duplicated(["qcap", "seed"]).any():
        raise ValueError("duplicate qcap/seed keys")
    if frame[list(required)].isna().any().any():
        raise ValueError("missing values in required fields")
    cert = frame["certificate_ok"].astype(str).str.lower().isin({"true", "1"})
    if not cert.all():
        raise ValueError("not every strict Qmax run is certified")
    if set(frame["policy"]) != {"FreshSky-Safe"} or set(frame["extra_delay"]) != {4}:
        raise ValueError("prototype expects FreshSky-Safe at d=4")


def seed_pressure(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for _, row in frame.iterrows():
        qcap = float(row["qcap"])
        queues = parse_vector(row["uav_max_queue"], "uav_max_queue")
        powers = parse_vector(row["uav_avg_power"], "uav_avg_power")
        rows.append(
            {
                "qcap": qcap,
                "seed": int(row["seed"]),
                "rejection_pct": 100.0 * float(row["rejection_rate"]),
                "uav_near_cap_pct": 100.0 * float(np.mean(queues >= 0.90 * qcap)),
                "median_queue_pressure_pct": 100.0 * float(np.median(queues / qcap)),
                "uav_near_budget_pct": 100.0
                * float(np.mean(powers >= 0.95 * PBAR_MW)),
                "bound_slack_mw": float(
                    row["finite_window_bound"] - row["max_avg_power"]
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize(
    frame: pd.DataFrame, metrics: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for cap, part in frame.groupby("qcap", sort=True):
        row: dict[str, float] = {
            "qcap": float(cap),
            "seeds": int(part["seed"].nunique()),
        }
        for metric in metrics:
            mean, half = mean_ci(part[metric])
            row[metric] = mean
            row[f"{metric}_ci95"] = half
        rows.append(row)
    return pd.DataFrame(rows).sort_values("qcap").reset_index(drop=True)


def paired_significant(
    frame: pd.DataFrame, metric: str, cap: float, reference: float = 4.0
) -> bool:
    if cap == reference:
        return False
    pivot = frame[frame["qcap"].isin([reference, cap])].pivot(
        index="seed", columns="qcap", values=metric
    )
    difference = pivot[cap] - pivot[reference]
    mean, half = mean_ci(difference)
    return bool(mean - half > 0 or mean + half < 0)


def panel_label(ax: mpl.axes.Axes, label: str, x: float, y: float) -> None:
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=BLACK,
    )


def add_grid(ax: mpl.axes.Axes, axis: str = "both") -> None:
    ax.grid(axis=axis, color="#E8EAED", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)


def make_figure(
    frame: pd.DataFrame,
    outdir: Path,
    derived: Path | None = None,
    stem: str = "fig_tmc_service_pareto",
) -> None:
    """Build the queue-cap atlas and export all plotted quantities."""
    derived = outdir if derived is None else derived
    outdir.mkdir(parents=True, exist_ok=True)
    derived.mkdir(parents=True, exist_ok=True)
    pressure = seed_pressure(frame)
    anchor_metrics = ["wAoI", "max_avg_power", "rejection_rate"]
    anchor = summarize(frame, anchor_metrics)

    pressure_metrics = [
        "rejection_pct",
        "uav_near_cap_pct",
        "median_queue_pressure_pct",
        "uav_near_budget_pct",
        "bound_slack_mw",
    ]
    pressure_summary = summarize(pressure, pressure_metrics)

    tail_specs = [
        ("worst_user_mean_aoi", "Worst-user mean", "{:.1f}", False),
        ("p95_user_mean_aoi", "User-mean p95", "{:.1f}", False),
        ("p95_slot_aoi", "Slot AoI p95", "{:.1f}", False),
        ("p99_slot_aoi", "Slot AoI p99", "{:.1f}", False),
        ("cvar95_slot_aoi", "Slot AoI CVaR95", "{:.1f}", False),
        ("slot_aoi_gt40", r"$P(A>40)$", "{:.1%}", False),
        ("slot_aoi_gt60", r"$P(A>60)$", "{:.1%}", False),
        ("starvation_p95", "Gap p95", "{:.1f}", False),
        ("starvation_max", "Gap maximum", "{:.0f}", False),
        ("worst_service_rate", "Worst service", "{:.1%}", True),
    ]
    tail_metrics = [item[0] for item in tail_specs]
    tail_summary = summarize(frame, tail_metrics)

    anchor.to_csv(
        derived / "tmc_v3_qcap_anchor_summary.csv", index=False
    )
    pressure.to_csv(
        derived / "tmc_v3_qcap_pressure_seeds.csv", index=False
    )
    pressure_summary.to_csv(
        derived / "tmc_v3_qcap_pressure_summary.csv", index=False
    )
    legacy_metrics = [
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
    summarize(frame, legacy_metrics).to_csv(
        derived / "tmc_advanced_qcap_summary.csv", index=False
    )

    tail_cells: list[dict[str, float | bool | str]] = []
    risk_rows = []
    raw_rows = []
    sig_rows = []
    for metric, label, _, reverse in tail_specs:
        values = tail_summary[metric].to_numpy(dtype=float)
        risk_source = -values if reverse else values
        span = float(np.max(risk_source) - np.min(risk_source))
        risk = (
            np.zeros_like(risk_source)
            if span <= 0
            else (risk_source - np.min(risk_source)) / span
        )
        risk_rows.append(risk)
        raw_rows.append(values)
        sig = []
        for cap, value, score in zip(EXPECTED_CAPS, values, risk):
            significant = paired_significant(frame, metric, cap)
            sig.append(significant)
            tail_cells.append(
                {
                    "metric": metric,
                    "label": label,
                    "qcap": cap,
                    "mean": value,
                    "relative_risk": score,
                    "paired_vs_q4_significant": significant,
                }
            )
        sig_rows.append(sig)
    pd.DataFrame(tail_cells).to_csv(
        derived / "tmc_v3_qcap_tail_cells.csv", index=False
    )

    configure_style()
    fig = plt.figure(figsize=(183 * MM, 106 * MM))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.34, 1.0],
        height_ratios=[0.78, 1.22],
        wspace=0.39,
        hspace=0.43,
    )
    ax_anchor = fig.add_subplot(grid[:, 0])
    ax_pressure = fig.add_subplot(grid[0, 1])
    ax_tail = fig.add_subplot(grid[1, 1])

    # (a) Safety-freshness path: the visual anchor.
    caps = anchor["qcap"].to_numpy(dtype=float)
    norm = Normalize(vmin=float(np.min(caps)), vmax=float(np.max(caps)))
    cmap = mpl.colormaps["cividis"]
    max_rejection = max(float(anchor["rejection_rate"].max()), 1e-12)
    ax_anchor.plot(
        anchor["max_avg_power"],
        anchor["wAoI"],
        color="#A7ABB0",
        linewidth=1.0,
        zorder=1,
    )
    offsets = {
        4.0: (5, 5),
        8.0: (5, -10),
        12.0: (5, 5),
        16.0: (5, -10),
        24.0: (-24, 12),
        28.0: (-24, -5),
        32.0: (8, 7),
        36.0: (-28, -20),
        48.0: (10, 15),
    }
    for _, row in anchor.iterrows():
        cap = float(row["qcap"])
        color = cmap(norm(cap))
        size = 28.0 + 105.0 * float(row["rejection_rate"]) / max_rejection
        ax_anchor.errorbar(
            row["max_avg_power"],
            row["wAoI"],
            xerr=row["max_avg_power_ci95"],
            yerr=row["wAoI_ci95"],
            fmt="none",
            ecolor=color,
            elinewidth=0.75,
            capsize=1.8,
            alpha=0.82,
            zorder=2,
        )
        ax_anchor.scatter(
            row["max_avg_power"],
            row["wAoI"],
            s=size,
            color=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )
        ax_anchor.annotate(
            f"{cap:g}",
            (row["max_avg_power"], row["wAoI"]),
            xytext=offsets[cap],
            textcoords="offset points",
            fontsize=5.7,
            fontweight="bold" if cap in {4.0, 24.0, 36.0, 48.0} else "normal",
            color=BLACK,
        )
    ax_anchor.axvline(PBAR_MW, color=BLACK, linestyle="--", linewidth=0.75)
    ax_anchor.text(
        PBAR_MW - 0.0007,
        0.985,
        "individual budget",
        transform=ax_anchor.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=5.6,
    )
    ax_anchor.text(
        0.025,
        0.025,
        "marker area = candidate rejection",
        transform=ax_anchor.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.4,
        color=DARK_GREY,
    )
    ax_anchor.set_xlabel("maximum individual average power (mW)")
    ax_anchor.set_ylabel("weighted AoI (slots)")
    ax_anchor.set_title(
        r"Nine-point safety--freshness path ($d=4$, 20 paired seeds)",
        pad=4,
    )
    add_grid(ax_anchor)
    panel_label(ax_anchor, "a", -0.14, 1.04)

    # (b) Constraint pressure. Every series is one value per seed before CI.
    pressure_styles = [
        ("rejection_pct", "candidate rejection", VERMILLION, "o"),
        ("uav_near_cap_pct", r"UAVs ever $\geq0.9Q_{\max}$", PURPLE, "s"),
        (
            "median_queue_pressure_pct",
            "median UAV queue / cap",
            BLUE,
            "D",
        ),
        (
            "uav_near_budget_pct",
            r"UAVs $\geq0.95\bar p$",
            GREEN,
            "^",
        ),
    ]
    for metric, label, color, marker in pressure_styles:
        ax_pressure.errorbar(
            pressure_summary["qcap"],
            pressure_summary[metric],
            yerr=pressure_summary[f"{metric}_ci95"],
            color=color,
            marker=marker,
            markersize=3.3,
            linewidth=0.9,
            elinewidth=0.65,
            capsize=1.5,
            label=label,
            zorder=3,
        )
    ax_pressure.set_xticks(EXPECTED_CAPS)
    ax_pressure.set_ylim(-3, 104)
    ax_pressure.set_xlabel(r"queue cap $Q_{\max}$")
    ax_pressure.set_ylabel("seed-level pressure (%)")
    ax_pressure.legend(
        loc="upper right",
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=0.7,
        labelspacing=0.25,
    )
    add_grid(ax_pressure)
    panel_label(ax_pressure, "b", -0.18, 1.08)

    # (c) Full tail-risk matrix. Color is within-row relative risk; text is raw.
    risk_matrix = np.asarray(risk_rows, dtype=float)
    raw_matrix = np.asarray(raw_rows, dtype=float)
    sig_matrix = np.asarray(sig_rows, dtype=bool)
    risk_cmap = LinearSegmentedColormap.from_list(
        "risk", ["#F7F7F7", "#F5C2A4", "#D55E00", "#7F1D13"]
    )
    image = ax_tail.imshow(
        risk_matrix,
        aspect="auto",
        cmap=risk_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    for i, (_, _, formatter, _) in enumerate(tail_specs):
        for j, cap in enumerate(EXPECTED_CAPS):
            score = risk_matrix[i, j]
            color = "white" if score > 0.62 else BLACK
            text = formatter.format(raw_matrix[i, j])
            if sig_matrix[i, j]:
                text += "*"
            ax_tail.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=4.5,
                color=color,
                fontweight="bold" if sig_matrix[i, j] else "normal",
            )
    ax_tail.set_xticks(range(len(EXPECTED_CAPS)), [f"{x:g}" for x in EXPECTED_CAPS])
    ax_tail.set_yticks(
        range(len(tail_specs)), [item[1] for item in tail_specs]
    )
    ax_tail.set_xlabel(r"queue cap $Q_{\max}$")
    ax_tail.tick_params(axis="both", length=0)
    for spine in ax_tail.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax_tail, fraction=0.035, pad=0.025)
    cbar.set_label("within-metric relative risk", labelpad=2)
    cbar.ax.tick_params(labelsize=4.8, length=1.8)
    ax_tail.text(
        1.0,
        1.025,
        "* paired 95% CI vs $Q_{\\max}=4$ excludes 0",
        transform=ax_tail.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.8,
        color=DARK_GREY,
    )
    panel_label(ax_tail, "c", -0.18, 1.04)

    fig.subplots_adjust(left=0.078, right=0.978, top=0.94, bottom=0.095)
    target = outdir / stem
    fig.savefig(target.with_suffix(".pdf"))
    fig.savefig(target.with_suffix(".svg"))
    fig.savefig(target.with_suffix(".png"), dpi=450)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_input = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "results"
        / "tmc_extended"
        / "tmc_qcap_service_seeds.csv"
    )
    default_output = Path(__file__).resolve().parent / "out"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    validate(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_figure(frame, args.output_dir, stem="fig6_v3")
    print(f"validated rows={len(frame)}, caps={EXPECTED_CAPS}, seeds/cap=20")
    for suffix in (".pdf", ".svg", ".png"):
        path = args.output_dir / f"fig6_v3{suffix}"
        print(f"saved {path} ({path.stat().st_size} bytes)")
    print("saved auditable anchor, pressure-seed, pressure-summary, and tail-cell CSVs")


if __name__ == "__main__":
    main()
