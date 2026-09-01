# -*- coding: utf-8 -*-
"""Expected receiver-AoI tracking under delayed ACK/NACK.

The expected-AoI variant follows the information pattern of Ji et al.,
IEEE TCOM 2024 (DOI: 10.1109/TCOMM.2024.3367712): scheduling uses an
estimate of receiver AoI formed from delayed feedback.  This is a matched
multi-UAV adaptation, not an exact reproduction of their CPOMDP policy.

For pending attempts, the estimator uses the success probability available
at each transmission time and computes the expected current AoI under a
conditional-independence approximation.  The channel posterior remains the
finite-grid Bayesian learner from Bayes-FreshSky.  All implementable variants
use the same Qmax safety shield.
"""
import os
import numpy as np
import pandas as pd

import belief_whittle as B
import bayes_hmm_experiment as E
import bayes_hmm_delayed_ack as D


DELAYS = D.DELAYS
SEEDS = D.SEEDS


class ConservativeBayesSafe(D.DelayAwareBayesSafe):
    name = "Conservative-AoI Bayes-Safe"
    age_mode = "conservative"

    def success_probability(self):
        belief = self.decision_belief
        return belief * B.PHI_LOS + (1.0 - belief) * B.PHI_NLOS


class ExpectedAoIBayesSafe(ConservativeBayesSafe):
    name = "Expected-AoI Bayes-Safe"
    age_mode = "expected"


class PerfectAoIBayesSafe(ConservativeBayesSafe):
    name = "Perfect-AoI Bayes-Safe"
    age_mode = "perfect"


class ExpectedAoIOracleSafe(D.DelayOracleSafe):
    name = "Expected-AoI fitted-GE-Safe"
    age_mode = "expected"

    def success_probability(self):
        belief = self.filter.current(self.horizon)
        return belief * B.PHI_LOS + (1.0 - belief) * B.PHI_NLOS


POLICIES = (
    ConservativeBayesSafe,
    ExpectedAoIBayesSafe,
    PerfectAoIBayesSafe,
    ExpectedAoIOracleSafe,
)


def expected_pending_age(
    conservative_age, attempted_history, success_prob_history, t, delay
):
    """Expected AoI at decision t from unobserved attempts.

    ``conservative_age`` is the current AoI if every still-pending attempt
    failed.  The newest successful pending attempt determines current AoI.
    """
    expected = np.zeros_like(conservative_age, dtype=float)
    no_newer_success = np.ones_like(conservative_age, dtype=float)
    first_pending = max(0, t - delay)
    for tau in range(t - 1, first_pending - 1, -1):
        attempted = attempted_history[tau]
        prob = np.where(attempted, success_prob_history[tau], 0.0)
        age_if_newest_success = float(max(1, t - tau))
        expected += no_newer_success * prob * age_if_newest_success
        no_newer_success *= 1.0 - prob
    expected += no_newer_success * conservative_age
    return np.maximum(expected, 1.0)


def run_one(policy_class, cfg, extra_delay, seed):
    rng = np.random.default_rng(23000 + seed)
    n_uav = cfg.N
    policy = policy_class(cfg)
    pi_los, pth = B.geometry(cfg, rng)
    p_nl = np.full(n_uav, 1.0 / cfg.mean_dwell)
    p_ln = np.clip(p_nl * (1.0 - pi_los) / pi_los, 0.0, 1.0)
    p_ll = 1.0 - p_ln
    if hasattr(policy, "configure_channel"):
        policy.configure_channel(pi_los, p_ll, p_nl)

    los = rng.random(n_uav) < pi_los
    actual_age = np.ones(n_uav)
    conservative_age = np.ones(n_uav)
    queue = np.zeros(n_uav)
    event = rng.random(n_uav) < cfg.p_event
    attempted_history = []
    ack_history = []
    success_prob_history = []

    weighted_age = 0.0
    estimation_abs_error = 0.0
    estimation_bias = 0.0
    energy_sum = np.zeros(n_uav)
    queue_max = np.zeros(n_uav)
    count = 0

    for t in range(cfg.T):
        policy.horizon = min(t, extra_delay)
        expected_age = expected_pending_age(
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
        energy = np.zeros(n_uav)
        attempted = np.zeros(n_uav, dtype=bool)
        ack = np.zeros(n_uav, dtype=bool)
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

        u = rng.random(n_uav)
        los = np.where(los, u >= p_ln, u < p_nl)
        u = rng.random(n_uav)
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
        policy=policy.name,
        age_mode=policy.age_mode,
        extra_delay=extra_delay,
        seed=seed,
        wAoI=weighted_age / count,
        age_mae=estimation_abs_error / count,
        age_bias=estimation_bias / count,
        mean_power=float(np.mean(average_power)),
        max_avg_power=float(np.max(average_power)),
        max_queue=float(np.max(queue_max)),
        finite_window_bound=finite_window_bound,
        bound_ok=bool(np.max(average_power) <= finite_window_bound + 1e-9),
        certificate_ok=bool(np.max(queue_max) <= cfg.qcap + 1e-9),
    )


def main():
    cfg = E.clone_cfg(T=4000, pbar=0.4, qcap=24.0, mean_dwell=8.0)
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
            wAoI=("wAoI", "mean"),
            wAoI_std=("wAoI", "std"),
            age_mae=("age_mae", "mean"),
            age_bias=("age_bias", "mean"),
            max_avg_power=("max_avg_power", "mean"),
            max_queue=("max_queue", "max"),
            finite_window_bound=("finite_window_bound", "first"),
            bound_all_seeds=("bound_ok", "all"),
            certificate_all_seeds=("certificate_ok", "all"),
        )
    )
    os.makedirs(B.RES, exist_ok=True)
    summary_path = os.path.join(
        B.RES, "bayes_hmm_delayed_aoi_belief_summary.csv"
    )
    seed_path = os.path.join(
        B.RES, "bayes_hmm_delayed_aoi_belief_seeds.csv"
    )
    summary.to_csv(summary_path, index=False)
    seeds.to_csv(seed_path, index=False)
    print(summary.to_string(index=False))
    print(f"saved {summary_path} and {seed_path}")


if __name__ == "__main__":
    main()
