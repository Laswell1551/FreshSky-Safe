#!/usr/bin/env python3
"""Reproducible IoTJ-facing inference evidence for FreshSky-Safe.

The script keeps the manuscript's hidden Gilbert--Elliott channel, delayed
ACK/NACK semantics, Bayesian model grid, and physical delivery model.  It adds
an open-loop probing protocol so prediction losses are compared on identical
observations, independently of a scheduler's chosen actions.

Outputs
-------
iotj_evidence/stationary_prediction_seeds.csv
iotj_evidence/stationary_prediction_summary.csv
iotj_evidence/shift_prediction_windows.csv
iotj_evidence/shift_prediction_summary.csv
iotj_evidence/iotj_inference_evidence_summary.md
figures/fig_iotj_inference.{pdf,png,svg}
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
EXPERIMENTS = PROJECT / "experiments"
RESULTS = PROJECT / "iotj_evidence"
FIGURES = PROJECT / "figures"
sys.path.insert(0, str(EXPERIMENTS))

import belief_whittle as B  # noqa: E402
import bayes_hmm_experiment as E  # noqa: E402
import bayes_hmm_delayed_ack as D  # noqa: E402


DELAYS = (2, 4, 8)
DWELLS = (2.0, 8.0, 14.0)
SEEDS = tuple(range(20))
T = 4000
WARMUP = 300
SHIFT_SLOT = 2000
WINDOW = 100
FORGET = 0.002

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green": "#2E8B57",
    "red": "#B64342",
    "orange": "#D97706",
    "neutral": "#767676",
    "light_neutral": "#CFCECE",
}


def transition_arrays(pi_los: np.ndarray, dwell: float) -> tuple[np.ndarray, np.ndarray]:
    """Return P(L|L) and P(L|N) under the manuscript parameterization."""
    p_nl = np.full_like(pi_los, 1.0 / float(dwell), dtype=float)
    p_ln = np.clip(p_nl * (1.0 - pi_los) / pi_los, 0.0, 1.0)
    return 1.0 - p_ln, p_nl


def success_probability(belief: np.ndarray) -> np.ndarray:
    belief = np.asarray(belief, dtype=float)
    return belief * B.PHI_LOS + (1.0 - belief) * B.PHI_NLOS


def bayes_decision_belief(model: E.BayesianGEFilter, horizon: int) -> np.ndarray:
    theta = D.propagate(
        model.theta,
        model.pLL[None, :],
        model.pNL[None, :],
        horizon,
    )
    return np.sum(model.weight * theta, axis=1)


def posterior_correlation(model: E.BayesianGEFilter) -> np.ndarray:
    correlations = model.pLL - model.pNL
    return np.sum(model.weight * correlations[None, :], axis=1)


def round_robin_attempts(t: int, n_uav: int, capacity: int) -> np.ndarray:
    attempted = np.zeros(n_uav, dtype=bool)
    start = (t * capacity) % n_uav
    attempted[[(start + offset) % n_uav for offset in range(capacity)]] = True
    return attempted


def bernoulli_nll(probability: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-9, 1.0 - 1e-9)
    y = np.asarray(outcome, dtype=float)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def ci95(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return float("nan")
    critical = {10: 2.2621571627409915, 20: 2.093024054408263}.get(
        len(arr), 1.96
    )
    return float(critical * np.std(arr, ddof=1) / math.sqrt(len(arr)))


def stationary_run(dwell: float, delay: int, seed: int) -> dict[str, float]:
    """Evaluate predictors on one common exogenous probe stream."""
    seed_sequence = np.random.SeedSequence(71000 + 1000 * int(dwell) + seed)
    geometry_rng, initial_rng, ack_rng, channel_rng = [
        np.random.default_rng(child) for child in seed_sequence.spawn(4)
    ]
    cfg = E.clone_cfg(T=T, warmup=WARMUP, mean_dwell=dwell)
    pi_los, _ = B.geometry(cfg, geometry_rng)
    p_ll, p_nl = transition_arrays(pi_los, dwell)

    bayes = E.BayesianGEFilter(cfg.N, forget=FORGET)
    known = D.KnownTransitionFilter(pi_los, p_ll, p_nl)
    static_q = success_probability(pi_los)
    los = initial_rng.random(cfg.N) < pi_los
    attempted_history: list[np.ndarray] = []
    ack_history: list[np.ndarray] = []

    brier = {name: [] for name in ("bayesian", "known_law", "static_geometry")}
    nll = {name: [] for name in brier}
    correlation_error: list[float] = []
    entropy: list[float] = []
    true_correlation = p_ll - p_nl

    for t in range(cfg.T):
        horizon = min(t, delay)
        q_bayes = success_probability(bayes_decision_belief(bayes, horizon))
        q_known = success_probability(known.current(horizon))
        predictions = {
            "bayesian": q_bayes,
            "known_law": q_known,
            "static_geometry": static_q,
        }

        attempted = round_robin_attempts(t, cfg.N, cfg.M)
        ack = attempted & (
            ack_rng.random(cfg.N)
            < np.where(los, B.PHI_LOS, B.PHI_NLOS)
        )
        attempted_history.append(attempted)
        ack_history.append(ack)

        if t >= cfg.warmup:
            for name, prediction in predictions.items():
                q = prediction[attempted]
                y = ack[attempted].astype(float)
                brier[name].extend(((q - y) ** 2).tolist())
                nll[name].extend(bernoulli_nll(q, y).tolist())
            correlation_error.append(
                float(np.mean(np.abs(posterior_correlation(bayes) - true_correlation)))
            )
            entropy.append(float(np.mean(bayes.normalized_entropy)))

        feedback_slot = t - delay
        if feedback_slot >= 0:
            old_attempted = attempted_history[feedback_slot]
            old_ack = ack_history[feedback_slot]
            bayes.step(old_attempted, old_ack)
            known.step(old_attempted, old_ack)

        transition_draw = channel_rng.random(cfg.N)
        los = np.where(los, transition_draw >= (1.0 - p_ll), transition_draw < p_nl)

    row: dict[str, float] = {
        "dwell": dwell,
        "delay": delay,
        "seed": seed,
        "observations": len(brier["bayesian"]),
        "posterior_correlation_mae": float(np.mean(correlation_error)),
        "posterior_entropy": float(np.mean(entropy)),
    }
    for name in brier:
        row[f"brier_{name}"] = float(np.mean(brier[name]))
        row[f"nll_{name}"] = float(np.mean(nll[name]))
    row["brier_skill_vs_static_pct"] = 100.0 * (
        row["brier_static_geometry"] - row["brier_bayesian"]
    ) / row["brier_static_geometry"]
    row["brier_gap_to_known_pct"] = 100.0 * (
        row["brier_bayesian"] - row["brier_known_law"]
    ) / row["brier_known_law"]
    return row


def summarize_stationary(seeds: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "brier_bayesian",
        "brier_known_law",
        "brier_static_geometry",
        "nll_bayesian",
        "nll_known_law",
        "nll_static_geometry",
        "brier_skill_vs_static_pct",
        "brier_gap_to_known_pct",
        "posterior_correlation_mae",
        "posterior_entropy",
    )
    rows = []
    for (dwell, delay), part in seeds.groupby(["dwell", "delay"]):
        row: dict[str, float] = {
            "dwell": dwell,
            "delay": delay,
            "seeds": part.seed.nunique(),
        }
        for metric in metrics:
            row[metric] = float(part[metric].mean())
            row[f"{metric}_ci95"] = ci95(part[metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dwell", "delay"])


def shift_run(delay: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Track prediction and posterior response to dwell 2 -> 14."""
    seed_sequence = np.random.SeedSequence(91000 + seed)
    geometry_rng, initial_rng, ack_rng, channel_rng = [
        np.random.default_rng(child) for child in seed_sequence.spawn(4)
    ]
    cfg = E.clone_cfg(T=T, warmup=WARMUP, mean_dwell=2.0)
    pi_los, _ = B.geometry(cfg, geometry_rng)
    pre_ll, pre_nl = transition_arrays(pi_los, 2.0)
    post_ll, post_nl = transition_arrays(pi_los, 14.0)
    bayes = E.BayesianGEFilter(cfg.N, forget=FORGET)
    fixed_pre = D.KnownTransitionFilter(pi_los, pre_ll, pre_nl)
    static_q = success_probability(pi_los)
    los = initial_rng.random(cfg.N) < pi_los
    attempted_history: list[np.ndarray] = []
    ack_history: list[np.ndarray] = []

    loss_records = []
    correlation_records = []
    for t in range(cfg.T):
        horizon = min(t, delay)
        q_bayes = success_probability(bayes_decision_belief(bayes, horizon))
        q_fixed = success_probability(fixed_pre.current(horizon))
        attempted = round_robin_attempts(t, cfg.N, cfg.M)
        ack = attempted & (
            ack_rng.random(cfg.N)
            < np.where(los, B.PHI_LOS, B.PHI_NLOS)
        )
        attempted_history.append(attempted)
        ack_history.append(ack)

        if t >= cfg.warmup:
            window_start = (t // WINDOW) * WINDOW
            y = ack[attempted].astype(float)
            for method, prediction in (
                ("Bayesian ensemble", q_bayes),
                ("Fixed pre-shift law", q_fixed),
                ("Static geometry", static_q),
            ):
                q = prediction[attempted]
                loss_records.append(
                    {
                        "seed": seed,
                        "delay": delay,
                        "window_start": window_start,
                        "method": method,
                        "squared_error_sum": float(np.sum((q - y) ** 2)),
                        "observations": len(y),
                    }
                )
            true_corr = (pre_ll - pre_nl) if t < SHIFT_SLOT else (post_ll - post_nl)
            correlation_records.append(
                {
                    "seed": seed,
                    "delay": delay,
                    "window_start": window_start,
                    "posterior_correlation": float(np.mean(posterior_correlation(bayes))),
                    "true_correlation": float(np.mean(true_corr)),
                }
            )

        feedback_slot = t - delay
        if feedback_slot >= 0:
            old_attempted = attempted_history[feedback_slot]
            old_ack = ack_history[feedback_slot]
            bayes.step(old_attempted, old_ack)
            fixed_pre.step(old_attempted, old_ack)

        p_ll, p_nl = (pre_ll, pre_nl) if t < SHIFT_SLOT else (post_ll, post_nl)
        transition_draw = channel_rng.random(cfg.N)
        los = np.where(los, transition_draw >= (1.0 - p_ll), transition_draw < p_nl)

    losses = pd.DataFrame(loss_records)
    losses = (
        losses.groupby(
            ["seed", "delay", "window_start", "method"], as_index=False
        )[["squared_error_sum", "observations"]]
        .sum()
    )
    losses["brier"] = losses.squared_error_sum / losses.observations
    losses["window_center"] = losses.window_start + WINDOW / 2.0

    correlations = pd.DataFrame(correlation_records)
    correlations = correlations.groupby(
        ["seed", "delay", "window_start"], as_index=False
    )[["posterior_correlation", "true_correlation"]].mean()
    correlations["window_center"] = correlations.window_start + WINDOW / 2.0
    return losses, correlations


def shift_phase(window_center: float) -> str | None:
    if window_center < SHIFT_SLOT:
        return "pre"
    if window_center < SHIFT_SLOT + 500:
        return "post_early"
    if window_center >= SHIFT_SLOT + 1000:
        return "post_late"
    return None


def summarize_shift(losses: pd.DataFrame, correlations: pd.DataFrame) -> pd.DataFrame:
    losses = losses.copy()
    losses["phase"] = losses.window_center.map(shift_phase)
    phase_seed = (
        losses.dropna(subset=["phase"])
        .groupby(["seed", "delay", "phase", "method"], as_index=False)
        .apply(
            lambda frame: pd.Series(
                {
                    "brier": float(
                        frame.squared_error_sum.sum() / frame.observations.sum()
                    )
                }
            ),
            include_groups=False,
        )
    )
    rows = []
    for (delay, phase, method), part in phase_seed.groupby(
        ["delay", "phase", "method"]
    ):
        rows.append(
            {
                "metric": "brier",
                "delay": delay,
                "phase": phase,
                "method": method,
                "mean": float(part.brier.mean()),
                "ci95": ci95(part.brier),
                "seeds": part.seed.nunique(),
            }
        )

    pivot = phase_seed.pivot_table(
        index=["seed", "delay", "phase"], columns="method", values="brier"
    ).reset_index()
    pivot["bayes_skill_vs_fixed_pct"] = 100.0 * (
        pivot["Fixed pre-shift law"] - pivot["Bayesian ensemble"]
    ) / pivot["Fixed pre-shift law"]
    for (delay, phase), part in pivot.groupby(["delay", "phase"]):
        rows.append(
            {
                "metric": "bayes_skill_vs_fixed_pct",
                "delay": delay,
                "phase": phase,
                "method": "Bayesian ensemble",
                "mean": float(part.bayes_skill_vs_fixed_pct.mean()),
                "ci95": ci95(part.bayes_skill_vs_fixed_pct),
                "seeds": part.seed.nunique(),
            }
        )

    corr = correlations.copy()
    corr["phase"] = corr.window_center.map(shift_phase)
    corr["correlation_abs_error"] = np.abs(
        corr.posterior_correlation - corr.true_correlation
    )
    corr_seed = corr.dropna(subset=["phase"]).groupby(
        ["seed", "delay", "phase"], as_index=False
    ).correlation_abs_error.mean()
    for (delay, phase), part in corr_seed.groupby(["delay", "phase"]):
        rows.append(
            {
                "metric": "posterior_correlation_mae",
                "delay": delay,
                "phase": phase,
                "method": "Bayesian ensemble",
                "mean": float(part.correlation_abs_error.mean()),
                "ci95": ci95(part.correlation_abs_error),
                "seeds": part.seed.nunique(),
            }
        )
    return pd.DataFrame(rows)


def aggregate_trajectory(
    frame: pd.DataFrame, value: str, groups: list[str]
) -> pd.DataFrame:
    rows = []
    for key, part in frame.groupby(groups):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(groups, key))
        row["mean"] = float(part[value].mean())
        row["ci95"] = ci95(part[value])
        row["seeds"] = part.seed.nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def annotated_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    ci: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    cbar_label: str,
) -> None:
    extent = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))), 1.0)
    image = ax.imshow(matrix, cmap="RdBu", vmin=-extent, vmax=extent, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_xlabel("pending-feedback delay")
    ax.set_ylabel("mean NLoS dwell (slots)")
    ax.set_title(title, pad=8)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if abs(matrix[row, col]) > 0.58 * extent else "#222222"
            ax.text(
                col,
                row,
                f"{matrix[row, col]:.1f}\n±{ci[row, col]:.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)


def render_figure(
    stationary: pd.DataFrame,
    shift_losses: pd.DataFrame,
    shift_correlations: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.5,
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 6.9))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    skill = stationary.pivot(
        index="dwell", columns="delay", values="brier_skill_vs_static_pct"
    ).reindex(index=DWELLS, columns=DELAYS)
    skill_ci = stationary.pivot(
        index="dwell", columns="delay", values="brier_skill_vs_static_pct_ci95"
    ).reindex(index=DWELLS, columns=DELAYS)
    annotated_heatmap(
        ax_a,
        skill.to_numpy(),
        skill_ci.to_numpy(),
        [rf"$d={delay}$" for delay in DELAYS],
        [f"{int(dwell)}" for dwell in DWELLS],
        "Bayesian predictive skill over static geometry",
        "Brier-score reduction (%)",
    )

    correlation_trajectory = aggregate_trajectory(
        shift_correlations,
        "posterior_correlation",
        ["window_center"],
    )
    true_trajectory = aggregate_trajectory(
        shift_correlations,
        "true_correlation",
        ["window_center"],
    )
    x = correlation_trajectory.window_center.to_numpy()
    y = correlation_trajectory["mean"].to_numpy()
    band = correlation_trajectory.ci95.to_numpy()
    ax_b.plot(x, y, color=PALETTE["blue_main"], lw=2.1, label="Bayesian posterior mean")
    ax_b.fill_between(x, y - band, y + band, color=PALETTE["blue_secondary"], alpha=0.18)
    ax_b.plot(
        true_trajectory.window_center,
        true_trajectory["mean"],
        color="#222222",
        lw=1.8,
        ls="--",
        label="true mean correlation",
    )
    ax_b.axvline(SHIFT_SLOT, color=PALETTE["red"], lw=1.3, ls=":")
    ax_b.text(SHIFT_SLOT + 45, ax_b.get_ylim()[0] + 0.03, "dwell 2→14", color=PALETTE["red"])
    ax_b.set_xlabel("slot")
    ax_b.set_ylabel(r"transition correlation $p_{LL}-p_{NL}$")
    ax_b.set_title("Posterior response to an unannounced transition shift", pad=8)
    ax_b.legend(loc="lower right", fontsize=9)
    ax_b.grid(axis="y", color="#E5E5E5", lw=0.7)

    calibration_path = PROJECT / "experiments" / "results" / "tmc_aoi_calibration_paired.csv"
    calibration = pd.read_csv(calibration_path)
    gain = calibration.pivot(
        index="mean_dwell", columns="extra_delay", values="gain_vs_conservative_pct"
    ).reindex(index=(2.0, 4.0, 8.0, 14.0), columns=DELAYS)
    gain_ci = calibration.pivot(
        index="mean_dwell",
        columns="extra_delay",
        values="gain_vs_conservative_ci95_halfwidth",
    ).reindex(index=(2.0, 4.0, 8.0, 14.0), columns=DELAYS)
    annotated_heatmap(
        ax_c,
        gain.to_numpy(),
        gain_ci.to_numpy(),
        [rf"$d={delay}$" for delay in DELAYS],
        ["2", "4", "8", "14"],
        "Decision gain from expected receiver-AoI reasoning",
        "W-AoI reduction (%)",
    )

    closed_loop = pd.read_csv(RESULTS / "closed_loop_grid_paired.csv")
    action_skill = closed_loop.pivot(
        index="dwell",
        columns="delay",
        values="bayesian_wAoI_skill_vs_static_pct",
    ).reindex(index=DWELLS, columns=DELAYS)
    action_skill_ci = closed_loop.pivot(
        index="dwell",
        columns="delay",
        values="bayesian_wAoI_skill_vs_static_pct_ci95",
    ).reindex(index=DWELLS, columns=DELAYS)
    annotated_heatmap(
        ax_d,
        action_skill.to_numpy(),
        action_skill_ci.to_numpy(),
        [rf"$d={delay}$" for delay in DELAYS],
        [f"{int(dwell)}" for dwell in DWELLS],
        "Closed-loop value of the Bayesian belief",
        "W-AoI reduction over static geometry (%)",
    )

    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.ravel()):
        ax.text(
            -0.16,
            1.09,
            label,
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold",
            va="top",
        )
    fig.tight_layout(pad=1.2, w_pad=2.0, h_pad=2.2)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(
            FIGURES / f"fig_iotj_inference.{suffix}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def write_summary(
    stationary: pd.DataFrame,
    shift_summary: pd.DataFrame,
) -> None:
    stationary_skill = stationary[
        [
            "dwell",
            "delay",
            "brier_skill_vs_static_pct",
            "brier_skill_vs_static_pct_ci95",
            "brier_gap_to_known_pct",
            "brier_gap_to_known_pct_ci95",
        ]
    ]
    late = shift_summary[
        (shift_summary.metric == "bayes_skill_vs_fixed_pct")
        & (shift_summary.phase == "post_late")
    ]
    lines = [
        "# IoTJ inference evidence summary",
        "",
        "All prediction comparisons use the same round-robin probe decisions, hidden-channel realizations, and ACK draws. The seed, not an individual ACK, is the inferential unit.",
        "",
        "## Stationary common-probe calibration",
        "",
        "| NLoS dwell | delay | Bayesian Brier skill vs static geometry | Bayesian Brier gap to known-law filter |",
        "|---:|---:|---:|---:|",
    ]
    for row in stationary_skill.itertuples(index=False):
        lines.append(
            f"| {row.dwell:.0f} | {row.delay:.0f} | "
            f"{row.brier_skill_vs_static_pct:.2f}% ± {row.brier_skill_vs_static_pct_ci95:.2f}% | "
            f"{row.brier_gap_to_known_pct:.2f}% ± {row.brier_gap_to_known_pct_ci95:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Abrupt-shift late-phase result",
            "",
        ]
    )
    for row in late.itertuples(index=False):
        lines.append(
            f"At delay {int(row.delay)}, the Bayesian ensemble changes Brier score by "
            f"{row.mean:.2f}% ± {row.ci95:.2f}% relative to the filter that remains fixed at the pre-shift transition law."
        )
    lines.extend(
        [
            "",
            "The open-loop inference experiment diagnoses perception/reasoning only. It does not replace the closed-loop scheduler comparison and does not imply universal W-AoI dominance.",
            "",
        ]
    )
    (RESULTS / "iotj_inference_evidence_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stationary_seeds = pd.DataFrame(
        [
            stationary_run(dwell, delay, seed)
            for dwell in DWELLS
            for delay in DELAYS
            for seed in SEEDS
        ]
    )
    stationary_summary = summarize_stationary(stationary_seeds)
    stationary_seeds.to_csv(RESULTS / "stationary_prediction_seeds.csv", index=False)
    stationary_summary.to_csv(
        RESULTS / "stationary_prediction_summary.csv", index=False
    )

    loss_frames = []
    correlation_frames = []
    for seed in SEEDS:
        losses, correlations = shift_run(delay=4, seed=seed)
        loss_frames.append(losses)
        correlation_frames.append(correlations)
    shift_losses = pd.concat(loss_frames, ignore_index=True)
    shift_correlations = pd.concat(correlation_frames, ignore_index=True)
    shift_windows = shift_losses.merge(
        shift_correlations,
        on=["seed", "delay", "window_start", "window_center"],
        how="left",
    )
    shift_summary = summarize_shift(shift_losses, shift_correlations)
    shift_windows.to_csv(RESULTS / "shift_prediction_windows.csv", index=False)
    shift_summary.to_csv(RESULTS / "shift_prediction_summary.csv", index=False)

    render_figure(stationary_summary, shift_losses, shift_correlations)
    write_summary(stationary_summary, shift_summary)
    print(stationary_summary.to_string(index=False))
    print("\n=== Shift summary ===")
    print(shift_summary.to_string(index=False))


if __name__ == "__main__":
    main()
