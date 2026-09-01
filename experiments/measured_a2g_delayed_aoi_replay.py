# -*- coding: utf-8 -*-
"""Delayed receiver-AoI estimators on the measured urban A2G CIR replay."""
import os
import numpy as np
import pandas as pd

import belief_whittle as B
import bayes_hmm_experiment as E
import bayes_hmm_delayed_aoi_belief as A
import measured_a2g_replay as M


DELAYS = (4, 8)
QUANTILES = M.QUANTILES
SEEDS = M.SEEDS
POLICIES = (
    A.ConservativeBayesSafe,
    A.ExpectedAoIBayesSafe,
    A.PerfectAoIBayesSafe,
)


def run_one(policy_class, cfg, good, extra_delay, seed):
    rng = np.random.default_rng(33000 + seed)
    policy = policy_class(cfg)
    _, pth = B.geometry(cfg, rng)
    length = len(good)
    offsets = rng.integers(0, length, size=cfg.N)
    actual_age = np.ones(cfg.N)
    conservative_age = np.ones(cfg.N)
    queue = np.zeros(cfg.N)
    event = rng.random(cfg.N) < cfg.p_event
    attempted_history = []
    ack_history = []
    success_prob_history = []

    weighted_age = 0.0
    estimation_abs_error = 0.0
    estimation_bias = 0.0
    energy_sum = np.zeros(cfg.N)
    queue_max = np.zeros(cfg.N)
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
        for n in selected:
            energy[n] = pth[n]
            attempted[n] = True
            ack[n] = rng.random() < (
                B.PHI_LOS if state[n] else B.PHI_NLOS
            )
        attempted_history.append(attempted)
        ack_history.append(ack)
        success_prob_history.append(predicted_success)

        policy.update(energy)
        actual_age = np.where(
            ack, 1.0, np.minimum(actual_age + 1.0, cfg.Amax)
        )
        conservative_age = np.minimum(conservative_age + 1.0, cfg.Amax)
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

        u = rng.random(cfg.N)
        event = np.where(event, u >= cfg.ev_off, u < cfg.p_event)
        if t >= cfg.warmup:
            weighted_age += float(np.sum(value * actual_age))
            estimation_abs_error += current_abs_error
            estimation_bias += current_bias
            energy_sum += energy
            count += 1

    average_power = energy_sum / count
    finite_window_bound = cfg.pbar + cfg.qcap / count
    return dict(
        seed=seed,
        wAoI=weighted_age / count,
        age_mae=estimation_abs_error / count,
        age_bias=estimation_bias / count,
        max_avg_power=float(np.max(average_power)),
        max_queue=float(np.max(queue_max)),
        finite_window_bound=finite_window_bound,
        bound_ok=bool(np.max(average_power) <= finite_window_bound + 1e-9),
        certificate_ok=bool(np.max(queue_max) <= cfg.qcap + 1e-9),
    )


def main():
    power_db = M.load_received_power()
    cfg = E.clone_cfg(T=5000, warmup=500, pbar=0.4, qcap=24.0)
    seed_frames = []
    summaries = []
    for quantile in QUANTILES:
        threshold = float(np.quantile(power_db, quantile))
        good = power_db >= threshold
        for delay in DELAYS:
            for policy_class in POLICIES:
                frame = pd.DataFrame(
                    [
                        run_one(
                            policy_class, cfg, good, delay, seed
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
                    dict(
                        quantile=quantile,
                        threshold_db=threshold,
                        good_fraction=float(np.mean(good)),
                        extra_delay=delay,
                        policy=policy_class.name,
                        age_mode=policy_class.age_mode,
                        wAoI=frame.wAoI.mean(),
                        wAoI_std=frame.wAoI.std(ddof=1),
                        age_mae=frame.age_mae.mean(),
                        age_bias=frame.age_bias.mean(),
                        max_avg_power=frame.max_avg_power.mean(),
                        max_queue=frame.max_queue.max(),
                        bound_all_seeds=frame.bound_ok.all(),
                        certificate_all_seeds=frame.certificate_ok.all(),
                    )
                )

    os.makedirs(B.RES, exist_ok=True)
    summary_path = os.path.join(
        B.RES, "measured_a2g_delayed_aoi_summary.csv"
    )
    seed_path = os.path.join(
        B.RES, "measured_a2g_delayed_aoi_seeds.csv"
    )
    summary = pd.DataFrame(summaries)
    summary.to_csv(summary_path, index=False)
    pd.concat(seed_frames, ignore_index=True).to_csv(seed_path, index=False)
    print(summary.to_string(index=False))
    print(f"saved {summary_path} and {seed_path}")


if __name__ == "__main__":
    main()
