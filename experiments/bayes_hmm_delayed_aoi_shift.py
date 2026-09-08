# -*- coding: utf-8 -*-
"""Expected receiver-AoI tracking with delayed ACK under an abrupt shift."""
import os
import numpy as np
import pandas as pd

import belief_whittle as B
import bayes_hmm_experiment as E
import bayes_hmm_delayed_aoi_belief as A


T_SHIFT = 2000
DWELL_PRE = 2.0
DWELL_POST = 14.0
DELAYS = (2, 4, 8)
SEEDS = (0, 1, 2, 3, 4)
POLICIES = (
    A.ConservativeBayesSafe,
    A.ExpectedAoIBayesSafe,
    A.PerfectAoIBayesSafe,
)


def run_one(policy_class, cfg, extra_delay, seed):
    rng = np.random.default_rng(25000 + seed)
    policy = policy_class(cfg)
    pi_los, pth = B.geometry(cfg, rng)
    los = rng.random(cfg.N) < pi_los
    actual_age = np.ones(cfg.N)
    conservative_age = np.ones(cfg.N)
    queue = np.zeros(cfg.N)
    event = rng.random(cfg.N) < cfg.p_event
    attempted_history = []
    ack_history = []
    success_prob_history = []

    pre_weighted_age = 0.0
    post_weighted_age = 0.0
    pre_count = 0
    post_count = 0
    post_abs_error = 0.0
    post_bias = 0.0
    post_energy = np.zeros(cfg.N)
    queue_max = np.zeros(cfg.N)

    for t in range(cfg.T):
        dwell = DWELL_PRE if t < T_SHIFT else DWELL_POST
        p_nl = np.full(cfg.N, 1.0 / dwell)
        p_ln = np.clip(
            p_nl * (1.0 - pi_los) / pi_los, 0.0, 1.0
        )
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
        energy = np.zeros(cfg.N)
        attempted = np.zeros(cfg.N, dtype=bool)
        ack = np.zeros(cfg.N, dtype=bool)
        for n in selected:
            energy[n] = pth[n]
            attempted[n] = True
            ack[n] = rng.random() < (
                B.PHI_LOS if los[n] else B.PHI_NLOS
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

        u = rng.random(cfg.N)
        los = np.where(los, u >= p_ln, u < p_nl)
        u = rng.random(cfg.N)
        event = np.where(event, u >= cfg.ev_off, u < cfg.p_event)

        if cfg.warmup <= t < T_SHIFT:
            pre_weighted_age += float(np.sum(value * actual_age))
            pre_count += 1
        elif t >= T_SHIFT:
            post_weighted_age += float(np.sum(value * actual_age))
            post_abs_error += current_abs_error
            post_bias += current_bias
            post_energy += energy
            post_count += 1

    post_avg_power = post_energy / post_count
    finite_window_bound = cfg.pbar + cfg.qcap / post_count
    return dict(
        policy=policy.name,
        age_mode=policy.age_mode,
        extra_delay=extra_delay,
        seed=seed,
        pre_wAoI=pre_weighted_age / pre_count,
        post_wAoI=post_weighted_age / post_count,
        post_age_mae=post_abs_error / post_count,
        post_age_bias=post_bias / post_count,
        post_max_avg_power=float(np.max(post_avg_power)),
        max_queue=float(np.max(queue_max)),
        finite_window_bound=finite_window_bound,
        bound_ok=bool(
            np.max(post_avg_power) <= finite_window_bound + 1e-9
        ),
        certificate_ok=bool(np.max(queue_max) <= cfg.qcap + 1e-9),
    )


def main():
    cfg = E.clone_cfg(T=4000, pbar=0.4, qcap=24.0)
    rows = []
    for policy in POLICIES:
        for delay in DELAYS:
            rows.extend(
                run_one(policy, cfg, delay, seed) for seed in SEEDS
            )
    seeds = pd.DataFrame(rows)
    summary = (
        seeds.groupby(
            ["policy", "age_mode", "extra_delay"], as_index=False
        )
        .agg(
            pre_wAoI=("pre_wAoI", "mean"),
            post_wAoI=("post_wAoI", "mean"),
            post_wAoI_std=("post_wAoI", "std"),
            post_age_mae=("post_age_mae", "mean"),
            post_age_bias=("post_age_bias", "mean"),
            post_max_avg_power=("post_max_avg_power", "mean"),
            max_queue=("max_queue", "max"),
            finite_window_bound=("finite_window_bound", "first"),
            bound_all_seeds=("bound_ok", "all"),
            certificate_all_seeds=("certificate_ok", "all"),
        )
    )
    os.makedirs(B.RES, exist_ok=True)
    summary_path = os.path.join(
        B.RES, "bayes_hmm_delayed_aoi_shift_summary.csv"
    )
    seed_path = os.path.join(
        B.RES, "bayes_hmm_delayed_aoi_shift_seeds.csv"
    )
    summary.to_csv(summary_path, index=False)
    seeds.to_csv(seed_path, index=False)
    print(summary.to_string(index=False))
    print(f"saved {summary_path} and {seed_path}")


if __name__ == "__main__":
    main()
