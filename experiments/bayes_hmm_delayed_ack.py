# -*- coding: utf-8 -*-
"""Robustness to delayed ACK/NACK and delayed receiver-AoI knowledge.

``extra_delay=d`` means feedback generated in slot t becomes usable after d
additional decision epochs.  Thus d=0 reproduces the original next-slot ACK
availability.  The scheduler never receives the instantaneous success vector.
"""
import os
import numpy as np
import pandas as pd

import belief_whittle as B
import bayes_hmm_experiment as E


DELAYS = (0, 1, 2, 4, 8)
SEEDS = (0, 1, 2, 3, 4)


def propagate(theta, p_ll, p_nl, steps):
    """Exact multi-step prediction for a two-state Markov chain.

    The recurrence ``x <- p_nl + (p_ll-p_nl)*x`` has the closed form below.
    This makes delayed-belief propagation O(NK), rather than O(NK*delay),
    while remaining numerically identical to repeated one-step prediction.
    """
    steps = int(steps)
    if steps <= 0:
        return np.asarray(theta, dtype=float).copy()
    contraction = p_ll - p_nl
    denominator = 1.0 - contraction
    stationary = np.divide(
        p_nl,
        denominator,
        out=np.full_like(np.asarray(p_nl, dtype=float), 0.5),
        where=np.abs(denominator) > 1e-15,
    )
    return stationary + (theta - stationary) * np.power(contraction, steps)


class DelayAwareBayesSafe(E.BayesFreshSkySafe):
    """Bayesian learner that propagates its lagged posterior to decision time."""

    name = "Delay-aware Bayes-Safe"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.horizon = 0

    @property
    def decision_belief(self):
        theta = propagate(
            self.filter.theta,
            self.filter.pLL[None, :],
            self.filter.pNL[None, :],
            self.horizon,
        )
        return np.sum(self.filter.weight * theta, axis=1)

    def __call__(self, age, unused, queue, value, pth, cfg):
        del unused
        score = (
            cfg.V * value * age * self.decision_belief * B.PHI_LOS
            - queue * pth
        )
        next_q = np.maximum(queue - cfg.pbar, 0.0) + pth
        score = np.where(next_q <= self.qcap + 1e-12, score, -1e18)
        return B.topM(score, cfg)


class DelayNaiveBayesSafe(DelayAwareBayesSafe):
    """Ablation that treats the lagged posterior as if it were current."""

    name = "Delay-naive Bayes-Safe"

    @property
    def decision_belief(self):
        return self.filter.belief


class KnownTransitionFilter:
    def __init__(self, pi_los, p_ll, p_nl):
        self.theta = np.asarray(pi_los, dtype=float).copy()
        self.p_ll = np.asarray(p_ll, dtype=float)
        self.p_nl = np.asarray(p_nl, dtype=float)

    def step(self, attempted, ack):
        attempted = np.asarray(attempted, dtype=bool)
        ack = np.asarray(ack, dtype=bool)
        post = self.theta.copy()
        p_ack = self.theta * B.PHI_LOS + (1.0 - self.theta) * B.PHI_NLOS
        yes = attempted & ack
        no = attempted & ~ack
        post[yes] = self.theta[yes] * B.PHI_LOS / (p_ack[yes] + 1e-12)
        post[no] = (
            self.theta[no] * (1.0 - B.PHI_LOS)
            / (1.0 - p_ack[no] + 1e-12)
        )
        self.theta = post * self.p_ll + (1.0 - post) * self.p_nl

    def current(self, horizon):
        return propagate(self.theta, self.p_ll, self.p_nl, horizon)


class DelayOracleSafe:
    """Known-transition comparator with the same delayed observations/shield."""

    name = "Delay-aware oracle-Safe"

    def __init__(self, cfg):
        self.qcap = float(cfg.qcap)
        self.horizon = 0
        self.filter = None

    def configure_channel(self, pi_los, p_ll, p_nl):
        self.filter = KnownTransitionFilter(pi_los, p_ll, p_nl)

    def __call__(self, age, unused, queue, value, pth, cfg):
        del unused
        belief = self.filter.current(self.horizon)
        score = cfg.V * value * age * belief * B.PHI_LOS - queue * pth
        next_q = np.maximum(queue - cfg.pbar, 0.0) + pth
        score = np.where(next_q <= self.qcap + 1e-12, score, -1e18)
        return B.topM(score, cfg)

    def update(self, energy):
        del energy

    def observe_feedback(self, attempted, ack):
        self.filter.step(attempted, ack)


POLICIES = (
    DelayAwareBayesSafe,
    DelayNaiveBayesSafe,
    DelayOracleSafe,
)


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
    scheduler_age = np.ones(n_uav)
    queue = np.zeros(n_uav)
    event = rng.random(n_uav) < cfg.p_event
    attempted_history = []
    ack_history = []

    weighted_age = 0.0
    age_error = 0.0
    energy_sum = np.zeros(n_uav)
    queue_max = np.zeros(n_uav)
    count = 0

    for t in range(cfg.T):
        policy.horizon = min(t, extra_delay)
        value = np.where(event, cfg.wev, 1.0)
        selected = policy(
            scheduler_age, None, queue, value, pth, cfg
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

        policy.update(energy)
        actual_age = np.where(
            ack, 1.0, np.minimum(actual_age + 1.0, cfg.Amax)
        )
        scheduler_age = np.minimum(scheduler_age + 1.0, cfg.Amax)
        queue = np.maximum(queue - cfg.pbar, 0.0) + energy
        queue_max = np.maximum(queue_max, queue)

        feedback_slot = t - extra_delay
        if feedback_slot >= 0:
            old_attempted = attempted_history[feedback_slot]
            old_ack = ack_history[feedback_slot]
            policy.observe_feedback(old_attempted, old_ack)
            scheduler_age[old_ack] = np.minimum(
                scheduler_age[old_ack], extra_delay + 1.0
            )

        u = rng.random(n_uav)
        los = np.where(los, u >= p_ln, u < p_nl)
        u = rng.random(n_uav)
        event = np.where(event, u >= cfg.ev_off, u < cfg.p_event)

        if t >= cfg.warmup:
            weighted_age += float(np.sum(value * actual_age))
            age_error += float(np.mean(scheduler_age - actual_age))
            energy_sum += energy
            count += 1

    average_power = energy_sum / count
    finite_window_bound = cfg.pbar + cfg.qcap / count
    return dict(
        policy=policy.name,
        extra_delay=extra_delay,
        seed=seed,
        wAoI=weighted_age / count,
        mean_age_overestimate=age_error / count,
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
        seeds.groupby(["policy", "extra_delay"], as_index=False)
        .agg(
            wAoI=("wAoI", "mean"),
            wAoI_std=("wAoI", "std"),
            mean_age_overestimate=("mean_age_overestimate", "mean"),
            max_avg_power=("max_avg_power", "mean"),
            max_queue=("max_queue", "max"),
            finite_window_bound=("finite_window_bound", "first"),
            bound_all_seeds=("bound_ok", "all"),
            certificate_all_seeds=("certificate_ok", "all"),
        )
    )
    os.makedirs(B.RES, exist_ok=True)
    summary_path = os.path.join(B.RES, "bayes_hmm_delayed_ack_summary.csv")
    seed_path = os.path.join(B.RES, "bayes_hmm_delayed_ack_seeds.csv")
    summary.to_csv(summary_path, index=False)
    seeds.to_csv(seed_path, index=False)
    print(summary.to_string(index=False))
    print(f"saved {summary_path} and {seed_path}")


if __name__ == "__main__":
    main()
