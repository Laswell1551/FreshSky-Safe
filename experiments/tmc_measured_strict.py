# -*- coding: utf-8 -*-
"""20-seed strict-CRN delayed-AoI evaluation on measured A2G replay."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import tmc_extended_evidence as X


DELAYS = (4, 8)
SEEDS = tuple(range(20))


def run_one(B, E, A, policy_class, cfg, good, extra_delay, seed):
    sequence = np.random.SeedSequence(43000 + seed)
    geometry_rng, initial_rng, ack_rng, event_rng = [
        np.random.default_rng(child) for child in sequence.spawn(4)
    ]
    policy = policy_class(cfg)
    _, pth = B.geometry(cfg, geometry_rng)
    current = good[:-1]
    nxt = good[1:]
    p_ll = float(np.mean(nxt[current])) if np.any(current) else 0.5
    p_nl = float(np.mean(nxt[~current])) if np.any(~current) else 0.5
    replay_stationary = np.full(cfg.N, float(np.mean(good)))
    if hasattr(policy, "configure_channel"):
        policy.configure_channel(replay_stationary, p_ll, p_nl)
    length = len(good)
    offsets = initial_rng.integers(0, length, size=cfg.N)
    actual_age = np.ones(cfg.N)
    conservative_age = np.ones(cfg.N)
    queue = np.zeros(cfg.N)
    event = initial_rng.random(cfg.N) < cfg.p_event
    attempted_history = []
    ack_history = []
    success_prob_history = []

    weighted_age = 0.0
    estimation_abs_error = 0.0
    estimation_bias = 0.0
    energy_sum = np.zeros(cfg.N)
    queue_max = np.zeros(cfg.N)
    age_hist = np.zeros(int(cfg.Amax) + 1, dtype=np.int64)
    count = 0

    for t in range(cfg.T):
        policy.horizon = min(t, extra_delay)
        expected_age = A.expected_pending_age(
            conservative_age,
            attempted_history,
            success_prob_history,
            t,
            extra_delay,
        )
        if policy.age_mode == "perfect":
            decision_age = actual_age
        elif policy.age_mode == "expected":
            decision_age = expected_age
        else:
            decision_age = conservative_age
        current_abs_error = float(
            np.mean(np.abs(decision_age - actual_age))
        )
        current_bias = float(np.mean(decision_age - actual_age))

        value = np.where(event, cfg.wev, 1.0)
        selected = policy(decision_age, None, queue, value, pth, cfg)
        predicted_success = np.asarray(
            policy.success_probability(), dtype=float
        )
        state = good[(offsets + t) % length]
        energy = np.zeros(cfg.N)
        attempted = np.zeros(cfg.N, dtype=bool)
        ack = np.zeros(cfg.N, dtype=bool)
        ack_draw = ack_rng.random(cfg.N)
        for n in selected:
            energy[n] = pth[n]
            attempted[n] = True
            ack[n] = ack_draw[n] < (
                B.PHI_LOS if state[n] else B.PHI_NLOS
            )
        attempted_history.append(attempted)
        ack_history.append(ack)
        success_prob_history.append(predicted_success)

        policy.update(energy)
        actual_age = np.where(ack, 1.0, actual_age + 1.0)
        conservative_age = conservative_age + 1.0
        queue = np.maximum(queue - cfg.pbar, 0.0) + energy
        queue_max = np.maximum(queue_max, queue)

        feedback_slot = t - extra_delay
        if feedback_slot >= 0:
            old_attempted = attempted_history[feedback_slot]
            old_ack = ack_history[feedback_slot]
            policy.observe_feedback(old_attempted, old_ack)
            conservative_age[old_ack] = np.minimum(
                conservative_age[old_ack], extra_delay + 1.0
            )

        u = event_rng.random(cfg.N)
        event = np.where(event, u >= cfg.ev_off, u < cfg.p_event)
        if t >= cfg.warmup:
            weighted_age += float(np.sum(value * actual_age))
            estimation_abs_error += current_abs_error
            estimation_bias += current_bias
            energy_sum += energy
            age_hist += np.bincount(
                actual_age.astype(int), minlength=len(age_hist)
            )[: len(age_hist)]
            count += 1

    average_power = energy_sum / count
    finite_window_bound = cfg.pbar + cfg.qcap / count
    return {
        "seed": seed,
        "wAoI": weighted_age / count,
        "age_mae": estimation_abs_error / count,
        "age_bias": estimation_bias / count,
        "p95_slot_aoi": X.histogram_percentile(age_hist, 95),
        "p99_slot_aoi": X.histogram_percentile(age_hist, 99),
        "cvar95_slot_aoi": X.histogram_cvar(age_hist, 95),
        "max_avg_power": float(np.max(average_power)),
        "max_queue": float(np.max(queue_max)),
        "finite_window_bound": finite_window_bound,
        "bound_ok": bool(
            np.max(average_power) <= finite_window_bound + 1e-9
        ),
        "certificate_ok": bool(
            np.max(queue_max) <= cfg.qcap + 1e-9
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    B, E, D, A = X.load_project(args.project)
    import measured_a2g_replay as M

    power_db = M.load_received_power()
    cfg = E.clone_cfg(T=5000, warmup=500, pbar=0.4, qcap=24.0)
    OursClass, baseline_policies, _ = X.build_policy_classes(B, E, D)
    ConservativeClass = next(
        policy
        for policy in baseline_policies
        if policy.name == "Conservative-AoI Safe"
    )

    class ExpectedClass(OursClass):
        name = "Expected-AoI Bayes-Safe"

    class ConservativeClassNamed(ConservativeClass):
        name = "Conservative-AoI Bayes-Safe"

    class PerfectClass(OursClass):
        name = "Perfect-AoI Bayes-Safe"
        age_mode = "perfect"

    policies = (
        ConservativeClassNamed,
        ExpectedClass,
        PerfectClass,
    )
    seed_frames = []
    summaries = []
    for quantile in M.QUANTILES:
        threshold = float(np.quantile(power_db, quantile))
        good = power_db >= threshold
        for delay in DELAYS:
            for policy_class in policies:
                frame = pd.DataFrame(
                    [
                        run_one(
                            B,
                            E,
                            A,
                            policy_class,
                            cfg,
                            good,
                            delay,
                            seed,
                        )
                        for seed in SEEDS
                    ]
                )
                frame.insert(0, "policy", policy_class.name)
                frame.insert(0, "age_mode", policy_class.age_mode)
                frame.insert(0, "extra_delay", delay)
                frame.insert(0, "good_fraction", float(np.mean(good)))
                frame.insert(0, "quantile", quantile)
                seed_frames.append(frame)
                summaries.append(
                    {
                        "quantile": quantile,
                        "threshold_db": threshold,
                        "good_fraction": float(np.mean(good)),
                        "extra_delay": delay,
                        "policy": policy_class.name,
                        "age_mode": policy_class.age_mode,
                        "wAoI": frame["wAoI"].mean(),
                        "wAoI_std": frame["wAoI"].std(ddof=1),
                        "age_mae": frame["age_mae"].mean(),
                        "age_bias": frame["age_bias"].mean(),
                        "p99_slot_aoi": frame["p99_slot_aoi"].mean(),
                        "cvar95_slot_aoi": frame[
                            "cvar95_slot_aoi"
                        ].mean(),
                        "max_avg_power": frame["max_avg_power"].mean(),
                        "max_queue": frame["max_queue"].max(),
                        "bound_all_seeds": frame["bound_ok"].all(),
                        "certificate_all_seeds": frame[
                            "certificate_ok"
                        ].all(),
                        "seeds": frame["seed"].nunique(),
                    }
                )

    seeds = pd.concat(seed_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    expected = "Expected-AoI Bayes-Safe"
    conservative = "Conservative-AoI Bayes-Safe"
    perfect = "Perfect-AoI Bayes-Safe"
    pivot = seeds.pivot(
        index=["quantile", "good_fraction", "extra_delay", "seed"],
        columns="policy",
        values="wAoI",
    )
    paired_rows = []
    for key, part in pivot.groupby(level=[0, 1, 2]):
        quantile, fraction, delay = key
        part = part.droplevel([0, 1, 2])
        gain = 100.0 * (
            part[conservative] - part[expected]
        ) / part[conservative]
        gap = 100.0 * (part[expected] - part[perfect]) / part[perfect]
        gain_mean, gain_ci = X.mean_ci95(gain)
        gap_mean, gap_ci = X.mean_ci95(gap)
        paired_rows.append(
            {
                "quantile": quantile,
                "good_fraction": fraction,
                "extra_delay": delay,
                "expected_gain_vs_conservative_pct": gain_mean,
                "expected_gain_ci95_halfwidth": gain_ci,
                "expected_gap_vs_true_aoi_pct": gap_mean,
                "expected_gap_ci95_halfwidth": gap_ci,
                "paired_seeds": len(part),
            }
        )
    paired = pd.DataFrame(paired_rows)

    seeds.to_csv(
        args.output / "measured_a2g_delayed_aoi_seeds.csv", index=False
    )
    summary.to_csv(
        args.output / "measured_a2g_delayed_aoi_summary.csv", index=False
    )
    paired.to_csv(
        args.output / "measured_a2g_delayed_aoi_paired.csv", index=False
    )
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
