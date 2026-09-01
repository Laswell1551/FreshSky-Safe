# -*- coding: utf-8 -*-
"""Extended journal evidence for FreshSky-Safe.

This script keeps the delayed-feedback simulator, geometry, random streams,
energy accounting, and configuration of the existing project, but adds:

1. a broader matched baseline suite, including recent index policies;
2. a 2x2x2 mechanism ablation (adaptive/stationary channel belief,
   expected/conservative receiver age, shield on/off); and
3. per-UAV service diagnostics over the Qmax frontier.

All reported values are generated from executed simulations.  Literature
"style" policies are matched adaptations, not exact reproductions.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# Resolve defaults from the script location so the one-command runner is
# portable across machines and non-ASCII directory names.
DEFAULT_PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = DEFAULT_PROJECT / "experiments" / "results" / "tmc_extended"


def load_project(project: Path):
    exp_dir = project / "experiments"
    sys.path.insert(0, str(exp_dir))
    import belief_whittle as B
    import bayes_hmm_experiment as E
    import bayes_hmm_delayed_ack as D
    import bayes_hmm_delayed_aoi_belief as A

    return B, E, D, A


def jain_index(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    denom = len(values) * float(np.sum(values**2))
    if denom <= 0:
        return 0.0
    return float(np.sum(values) ** 2 / denom)


def histogram_percentile(hist: np.ndarray, percentile: float) -> float:
    target = percentile / 100.0 * float(np.sum(hist))
    if target <= 0:
        return float("nan")
    return float(np.searchsorted(np.cumsum(hist), target, side="left"))


def histogram_cvar(hist: np.ndarray, percentile: float) -> float:
    """Mean of the upper histogram tail, with a fractional boundary bin."""
    hist = np.asarray(hist, dtype=float)
    total = float(np.sum(hist))
    if total <= 0:
        return float("nan")
    cutoff = percentile / 100.0 * total
    cumulative = np.cumsum(hist)
    boundary = int(np.searchsorted(cumulative, cutoff, side="left"))
    before = float(cumulative[boundary - 1]) if boundary > 0 else 0.0
    boundary_tail = max(
        0.0, float(hist[boundary]) - (cutoff - before)
    )
    indices = np.arange(len(hist), dtype=float)
    tail_count = boundary_tail + float(np.sum(hist[boundary + 1 :]))
    tail_sum = (
        boundary * boundary_tail
        + float(np.sum(indices[boundary + 1 :] * hist[boundary + 1 :]))
    )
    return tail_sum / max(tail_count, 1e-12)


def serialize_vector(values: np.ndarray) -> str:
    return ";".join(f"{float(value):.10g}" for value in values)


def mean_ci95(values: pd.Series) -> tuple[float, float]:
    """Two-sided Student-t interval for the supported seed counts."""
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return mean, float("nan")
    critical = {
        10: 2.2621571627409915,
        20: 2.093024054408263,
    }.get(len(arr), 1.96)
    return mean, float(
        critical * np.std(arr, ddof=1) / math.sqrt(len(arr))
    )


def zhao_optimal_randomized_rates(
    alpha: np.ndarray,
    success_probability: np.ndarray,
    capacity: float,
) -> np.ndarray:
    """Solve Zhao--Kadota's optimal randomized-rate allocation.

    Their Theorem-7 calibration minimizes
    ``sum(alpha_i / (p_i * mu_i))`` subject to ``sum(mu_i)=capacity``
    and ``0 < mu_i <= 1``. Here ``p_i`` is the stationary end-to-end
    success probability after specializing their two-hop model to our
    single-hop uplink.
    """
    alpha = np.asarray(alpha, dtype=float)
    success = np.clip(
        np.asarray(success_probability, dtype=float), 1e-12, 1.0
    )
    if alpha.shape != success.shape:
        raise ValueError("alpha and success_probability must have one shape")
    if np.any(alpha <= 0):
        raise ValueError("all fixed Zhao weights must be positive")
    n_sources = len(alpha)
    if not 0 < capacity <= n_sources:
        raise ValueError("capacity must be in (0, number of sources]")
    if capacity >= n_sources:
        return np.ones(n_sources, dtype=float)

    vartheta = alpha / success
    lower = 0.0
    upper = float(np.max(vartheta))
    while float(np.sum(np.minimum(1.0, np.sqrt(vartheta / upper)))) > capacity:
        upper *= 4.0
    for _ in range(100):
        threshold = 0.5 * (lower + upper)
        rates = np.minimum(1.0, np.sqrt(vartheta / threshold))
        if float(np.sum(rates)) > capacity:
            lower = threshold
        else:
            upper = threshold
    return np.minimum(1.0, np.sqrt(vartheta / upper))


def zhao_mmse_maxweight_priority(
    beta: np.ndarray,
    source_success: np.ndarray,
    destination_success: np.ndarray,
    estimated_destination_age: np.ndarray,
    estimated_source_system_time: np.ndarray,
    forward_delay: np.ndarray,
) -> np.ndarray:
    """Zhao--Kadota (2025) MMSE-MaxWeight priority (their Eq. 26)."""
    return (
        np.asarray(beta, dtype=float)
        * np.asarray(source_success, dtype=float)
        * np.asarray(destination_success, dtype=float)
        * (
            np.asarray(estimated_destination_age, dtype=float)
            - np.asarray(estimated_source_system_time, dtype=float)
            - np.asarray(forward_delay, dtype=float)
        )
    )


def build_policy_classes(B, E, D):
    class PolicyBase:
        age_mode = "expected"
        shielded = True
        adaptive = True
        geometry_prior = False
        belief_weight = 1.0
        scheduling = "dpp"
        name = "policy"

        def __init__(self, cfg):
            self.cfg = cfg
            self.qcap = float(cfg.qcap)
            self.horizon = 0
            self.last_proposed = 0
            self.last_rejected = 0
            self.last_eligible = 0
            self.Z = 0.0
            if self.adaptive:
                self.filter = E.BayesianGEFilter(cfg.N)
            else:
                self.filter = None
                self.stationary = np.full(cfg.N, 0.5)

        def configure_channel(self, pi_los, p_ll, p_nl):
            del p_ll, p_nl
            self.stationary = np.asarray(pi_los, dtype=float).copy()
            if self.adaptive and self.geometry_prior:
                candidate_stationary = self.filter.pNL / (
                    1.0 - self.filter.pLL + self.filter.pNL
                )
                sigma = 0.10
                distance = (
                    candidate_stationary[None, :]
                    - self.stationary[:, None]
                ) / sigma
                weights = np.exp(-0.5 * distance**2) + 1e-12
                self.filter.weight = weights / np.sum(
                    weights, axis=1, keepdims=True
                )
                self.filter.theta = np.broadcast_to(
                    candidate_stationary[None, :],
                    self.filter.theta.shape,
                ).copy()

        @property
        def decision_belief(self):
            if not self.adaptive:
                return self.stationary
            theta = D.propagate(
                self.filter.theta,
                self.filter.pLL[None, :],
                self.filter.pNL[None, :],
                self.horizon,
            )
            learned = np.sum(self.filter.weight * theta, axis=1)
            return (
                self.belief_weight * learned
                + (1.0 - self.belief_weight) * self.stationary
            )

        def success_probability(self):
            belief = self.decision_belief
            return belief * B.PHI_LOS + (1.0 - belief) * B.PHI_NLOS

        def _select(self, raw_score, queue, pth, cfg):
            raw_score = np.asarray(raw_score, dtype=float)
            proposed = B.topM(raw_score, cfg)
            self.last_proposed = len(proposed)
            if self.shielded:
                next_q = np.maximum(queue - cfg.pbar, 0.0) + pth
                feasible = next_q <= self.qcap + 1e-12
                self.last_rejected = int(
                    sum(not bool(feasible[n]) for n in proposed)
                )
                self.last_eligible = int(np.sum(feasible & (raw_score > 0)))
                score = np.where(feasible, raw_score, -1e18)
                return B.topM(score, cfg)
            self.last_rejected = 0
            self.last_eligible = int(np.sum(raw_score > 0))
            return proposed

        def __call__(self, age, unused, queue, value, pth, cfg):
            del unused
            belief = self.decision_belief
            if self.scheduling == "ji":
                raw = value * age * self.success_probability()
            elif self.scheduling == "tripathi":
                # Belief plug-in adaptation of the unreliable-channel
                # weighted-AoI Whittle index in Tripathi--Modiano (ToN'24).
                # The original index assumes an i.i.d. success probability;
                # here the causal posterior-predictive probability is used.
                success = np.clip(
                    self.success_probability(), 1e-6, 1.0
                )
                raw = (
                    value
                    * success
                    * age
                    * (age + (2.0 - success) / success)
                    / 2.0
                )
            elif self.scheduling == "maxage":
                raw = value * age
            elif self.scheduling == "aggregate":
                raw = (
                    cfg.V * value * age * self.success_probability()
                    - self.Z * pth
                )
            else:
                raw = (
                    cfg.V
                    * value
                    * age
                    * self.success_probability()
                    - queue * pth
                )
            return self._select(raw, queue, pth, cfg)

        def update(self, energy):
            if self.scheduling == "aggregate":
                self.Z = max(
                    0.0,
                    self.Z
                    + float(np.sum(energy))
                    - self.cfg.N * self.cfg.pbar,
                )

        def observe_feedback(self, attempted, ack):
            if self.adaptive:
                self.filter.step(attempted, ack)

    def make_policy(
        class_name,
        display_name,
        *,
        adaptive=True,
        age_mode="expected",
        shielded=True,
        scheduling="dpp",
        geometry_prior=False,
        belief_weight=1.0,
    ):
        attrs = {
            "name": display_name,
            "adaptive": adaptive,
            "age_mode": age_mode,
            "shielded": shielded,
            "scheduling": scheduling,
            "geometry_prior": geometry_prior,
            "belief_weight": belief_weight,
        }
        return type(class_name, (PolicyBase,), attrs)

    AdaptiveExpectedShield = make_policy(
        "AdaptiveExpectedShield",
        "FreshSky-Safe",
        adaptive=True,
        age_mode="expected",
        shielded=True,
    )
    GeometryPriorExpectedShield = make_policy(
        "GeometryPriorExpectedShield",
        "Geometry-Prior Bayes-Safe",
        adaptive=True,
        age_mode="expected",
        shielded=True,
        geometry_prior=True,
    )
    Shrink25ExpectedShield = make_policy(
        "Shrink25ExpectedShield",
        "Regularized Bayes-Safe (alpha=.25)",
        adaptive=True,
        age_mode="expected",
        shielded=True,
        belief_weight=0.25,
    )
    Shrink50ExpectedShield = make_policy(
        "Shrink50ExpectedShield",
        "Regularized Bayes-Safe (alpha=.50)",
        adaptive=True,
        age_mode="expected",
        shielded=True,
        belief_weight=0.50,
    )
    Shrink75ExpectedShield = make_policy(
        "Shrink75ExpectedShield",
        "Regularized Bayes-Safe (alpha=.75)",
        adaptive=True,
        age_mode="expected",
        shielded=True,
        belief_weight=0.75,
    )
    AdaptiveConservativeShield = make_policy(
        "AdaptiveConservativeShield",
        "Conservative-AoI Safe",
        adaptive=True,
        age_mode="conservative",
        shielded=True,
    )
    StationaryExpectedShield = make_policy(
        "StationaryExpectedShield",
        "Static-Geometry Safe",
        adaptive=False,
        age_mode="expected",
        shielded=True,
    )
    JiStyle = make_policy(
        "JiStyle",
        "Ji'24-style Greedy-Safe",
        adaptive=True,
        age_mode="expected",
        shielded=True,
        scheduling="ji",
    )
    ZhuStyle = make_policy(
        "ZhuStyle",
        "Zhu'26-style Aggregate-DPP",
        adaptive=True,
        age_mode="expected",
        shielded=False,
        scheduling="aggregate",
    )
    MaxAgeSafe = make_policy(
        "MaxAgeSafe",
        "Max-Age-First Safe",
        adaptive=True,
        age_mode="expected",
        shielded=True,
        scheduling="maxage",
    )
    TripathiWhittleSafe = make_policy(
        "TripathiWhittleSafe",
        "Tripathi'24 Whittle-Safe",
        adaptive=True,
        age_mode="expected",
        shielded=True,
        scheduling="tripathi",
    )

    class ZhaoMMSEMaxWeightSafe(PolicyBase):
        """Theorem-beta Zhao--Kadota'25 MW-E adaptation plus shield.

        Generate-at-will makes the source system time zero and same-slot
        delivery makes the forward delay zero. We compute Zhao--Kadota's
        fixed Theorem-7 beta from stationary geometry, replace the paper's
        fixed success product by the common causal posterior prediction, and
        then apply the common individual-energy shield. Time-varying event
        value is deliberately excluded: inserting it directly as beta would
        collapse this row pathwise to Ji'24-style greedy.
        """

        name = "Zhao'25 MMSE-MW-Safe"
        adaptive = True
        age_mode = "expected"
        shielded = True
        scheduling = "zhao"

        def configure_channel(self, pi_los, p_ll, p_nl):
            super().configure_channel(pi_los, p_ll, p_nl)
            stationary_success = np.clip(
                np.asarray(pi_los, dtype=float) * B.PHI_LOS
                + (1.0 - np.asarray(pi_los, dtype=float)) * B.PHI_NLOS,
                1e-12,
                1.0,
            )
            alpha = np.ones(self.cfg.N, dtype=float)
            self.randomized_rate = zhao_optimal_randomized_rates(
                alpha, stationary_success, float(self.cfg.M)
            )
            self.beta = alpha / (
                stationary_success * self.randomized_rate
            )

        def __call__(self, age, unused, queue, value, pth, cfg):
            del unused, value
            raw = zhao_mmse_maxweight_priority(
                self.beta,
                self.success_probability(),
                np.ones(cfg.N, dtype=float),
                age,
                np.zeros(cfg.N, dtype=float),
                np.zeros(cfg.N, dtype=float),
            )
            return self._select(raw, queue, pth, cfg)

    class RoundRobinSafe(PolicyBase):
        name = "Round-Robin Safe"
        adaptive = True
        age_mode = "expected"
        shielded = True
        scheduling = "roundrobin"

        def __init__(self, cfg):
            super().__init__(cfg)
            self.pointer = 0

        def __call__(self, age, unused, queue, value, pth, cfg):
            del age, unused, value
            order = [
                (self.pointer + offset) % cfg.N for offset in range(cfg.N)
            ]
            proposed = order[: cfg.M]
            self.pointer = (self.pointer + cfg.M) % cfg.N
            next_q = np.maximum(queue - cfg.pbar, 0.0) + pth
            feasible = next_q <= self.qcap + 1e-12
            self.last_proposed = len(proposed)
            self.last_rejected = int(
                sum(not bool(feasible[n]) for n in proposed)
            )
            self.last_eligible = int(np.sum(feasible))
            # Preserve cyclic priority while backfilling rejected top-M
            # entries with later feasible UAVs, as the score-based shield
            # already does.
            selected = []
            for n in order:
                if feasible[n]:
                    selected.append(n)
                    if len(selected) == cfg.M:
                        break
            return selected

    class KnownTransitionSafe(PolicyBase):
        name = "Known-Transition Safe"
        adaptive = False
        age_mode = "expected"
        shielded = True

        def configure_channel(self, pi_los, p_ll, p_nl):
            self.filter = D.KnownTransitionFilter(pi_los, p_ll, p_nl)

        @property
        def decision_belief(self):
            return self.filter.current(self.horizon)

        def observe_feedback(self, attempted, ack):
            self.filter.step(attempted, ack)

    class WangPORMABSafe(PolicyBase):
        """Wang et al. 2026 Whittle-like index with the common safety shield.

        Wang's published PORMAB assumes known Markov transitions and the
        receiver age state.  This matched adaptation retains its known-
        transition advantage, uses the same delayed ACK/NACK observations and
        expected receiver-age estimate as the tested system, multiplies by the
        common event value, and applies the individual queue-cap shield.
        """

        name = "Wang'26 PORMAB-Safe"
        adaptive = False
        age_mode = "expected"
        shielded = True
        scheduling = "wang"

        def configure_channel(self, pi_los, p_ll, p_nl):
            self.filter = D.KnownTransitionFilter(pi_los, p_ll, p_nl)
            self.p_ll = np.asarray(p_ll, dtype=float)
            self.p_nl = np.asarray(p_nl, dtype=float)

        @property
        def decision_belief(self):
            return self.filter.current(self.horizon)

        def __call__(self, age, unused, queue, value, pth, cfg):
            del unused
            belief = np.clip(self.decision_belief, 1e-8, 1.0 - 1e-8)
            # Wang's notation uses alpha=P(L|L) and beta=P(N|N).
            beta = 1.0 - self.p_nl
            raw = value * B.wang_index(
                age, belief, self.p_ll, beta
            )
            return self._select(raw, queue, pth, cfg)

        def observe_feedback(self, attempted, ack):
            self.filter.step(attempted, ack)

    factorial = {}
    for adaptive in (False, True):
        for age_mode in ("conservative", "expected"):
            for shielded in (False, True):
                label = (
                    ("Bayesian" if adaptive else "Stationary")
                    + "-"
                    + ("Expected" if age_mode == "expected" else "Conservative")
                    + "-"
                    + ("Shield" if shielded else "NoShield")
                )
                factorial[(adaptive, age_mode, shielded)] = make_policy(
                    label.replace("-", ""),
                    label,
                    adaptive=adaptive,
                    age_mode=age_mode,
                    shielded=shielded,
                )

    baseline_policies = (
        AdaptiveExpectedShield,
        JiStyle,
        ZhaoMMSEMaxWeightSafe,
        ZhuStyle,
        KnownTransitionSafe,
        WangPORMABSafe,
        TripathiWhittleSafe,
        AdaptiveConservativeShield,
        StationaryExpectedShield,
        MaxAgeSafe,
        RoundRobinSafe,
    )
    return AdaptiveExpectedShield, baseline_policies, factorial


def run_one(B, E, A, policy_class, cfg, extra_delay, seed):
    """Matched delayed-feedback run with per-UAV service diagnostics."""
    seed_sequence = np.random.SeedSequence(23000 + seed)
    geometry_rng, initial_rng, ack_rng, channel_rng, event_rng = [
        np.random.default_rng(child)
        for child in seed_sequence.spawn(5)
    ]
    n_uav = cfg.N
    policy = policy_class(cfg)
    pi_los, pth = B.geometry(cfg, geometry_rng)
    p_nl = np.full(n_uav, 1.0 / cfg.mean_dwell)
    p_ln = np.clip(p_nl * (1.0 - pi_los) / pi_los, 0.0, 1.0)
    p_ll = 1.0 - p_ln
    if hasattr(policy, "configure_channel"):
        policy.configure_channel(pi_los, p_ll, p_nl)

    los = initial_rng.random(n_uav) < pi_los
    actual_age = np.ones(n_uav)
    conservative_age = np.ones(n_uav)
    queue = np.zeros(n_uav)
    event = initial_rng.random(n_uav) < cfg.p_event
    attempted_history = []
    ack_history = []
    success_prob_history = []

    weighted_age = 0.0
    age_sum = np.zeros(n_uav)
    energy_sum = np.zeros(n_uav)
    queue_max = np.zeros(n_uav)
    attempts = np.zeros(n_uav, dtype=int)
    successes = np.zeros(n_uav, dtype=int)
    last_success = np.full(n_uav, cfg.warmup - 1, dtype=int)
    gaps = []
    age_hist = np.zeros(int(cfg.Amax) + 1, dtype=np.int64)
    rejected = 0
    proposed = 0
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
        if policy.age_mode == "expected":
            decision_age = expected_age
        elif policy.age_mode == "perfect":
            decision_age = actual_age
        else:
            decision_age = conservative_age

        value = np.where(event, cfg.wev, 1.0)
        selected = policy(decision_age, None, queue, value, pth, cfg)
        predicted_success = np.asarray(
            policy.success_probability(), dtype=float
        )
        energy = np.zeros(n_uav)
        attempted = np.zeros(n_uav, dtype=bool)
        ack = np.zeros(n_uav, dtype=bool)
        ack_draw = ack_rng.random(n_uav)
        for n in selected:
            energy[n] = pth[n]
            attempted[n] = True
            ack[n] = ack_draw[n] < (
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

        u = channel_rng.random(n_uav)
        los = np.where(los, u >= p_ln, u < p_nl)
        u = event_rng.random(n_uav)
        event = np.where(event, u >= cfg.ev_off, u < cfg.p_event)

        if t >= cfg.warmup:
            weighted_age += float(np.sum(value * actual_age))
            age_sum += actual_age
            energy_sum += energy
            attempts += attempted.astype(int)
            successes += ack.astype(int)
            for n in np.flatnonzero(ack):
                gaps.append(int(t - last_success[n]))
                last_success[n] = t
            age_hist += np.bincount(
                actual_age.astype(int), minlength=len(age_hist)
            )[: len(age_hist)]
            rejected += int(getattr(policy, "last_rejected", 0))
            proposed += int(getattr(policy, "last_proposed", 0))
            count += 1

    gaps.extend((cfg.T - last_success).astype(int).tolist())
    average_power = energy_sum / count
    mean_age_uav = age_sum / count
    service_rate = successes / count
    finite_window_bound = cfg.pbar + cfg.qcap / count
    shielded = bool(getattr(policy, "shielded", False))
    return dict(
        policy=policy.name,
        extra_delay=extra_delay,
        seed=seed,
        wAoI=weighted_age / count,
        worst_user_mean_aoi=float(np.max(mean_age_uav)),
        p95_user_mean_aoi=float(np.percentile(mean_age_uav, 95)),
        p95_slot_aoi=histogram_percentile(age_hist, 95),
        p99_slot_aoi=histogram_percentile(age_hist, 99),
        cvar95_slot_aoi=histogram_cvar(age_hist, 95),
        slot_aoi_gt40=float(np.sum(age_hist[41:]) / np.sum(age_hist)),
        slot_aoi_gt60=float(np.sum(age_hist[61:]) / np.sum(age_hist)),
        mean_service_rate=float(np.mean(service_rate)),
        worst_service_rate=float(np.min(service_rate)),
        jain_service_fairness=jain_index(service_rate),
        starvation_p95=float(np.percentile(gaps, 95)),
        starvation_max=float(np.max(gaps)),
        rejection_rate=(float(rejected / proposed) if proposed else 0.0),
        rejected_candidates=rejected,
        proposed_candidates=proposed,
        mean_power=float(np.mean(average_power)),
        max_avg_power=float(np.max(average_power)),
        max_queue=float(np.max(queue_max)),
        finite_window_bound=finite_window_bound,
        bound_ok=bool(np.max(average_power) <= finite_window_bound + 1e-9),
        certificate_ok=bool(
            shielded and np.max(queue_max) <= cfg.qcap + 1e-9
        ),
        uav_pi_los=serialize_vector(pi_los),
        uav_pth=serialize_vector(pth),
        uav_mean_aoi=serialize_vector(mean_age_uav),
        uav_avg_power=serialize_vector(average_power),
        uav_service_rate=serialize_vector(service_rate),
        uav_max_queue=serialize_vector(queue_max),
    )


def run_shift_one(
    B,
    E,
    A,
    policy_class,
    cfg,
    extra_delay,
    seed,
    shift_slot=2000,
    dwell_pre=2.0,
    dwell_post=14.0,
):
    """Delayed-feedback transition-shift run under the same policy contract."""
    seed_sequence = np.random.SeedSequence(33000 + seed)
    geometry_rng, initial_rng, ack_rng, channel_rng, event_rng = [
        np.random.default_rng(child)
        for child in seed_sequence.spawn(5)
    ]
    n_uav = cfg.N
    policy = policy_class(cfg)
    pi_los, pth = B.geometry(cfg, geometry_rng)
    p_nl_pre = np.full(n_uav, 1.0 / dwell_pre)
    p_ln_pre = np.clip(
        p_nl_pre * (1.0 - pi_los) / pi_los, 0.0, 1.0
    )
    if hasattr(policy, "configure_channel"):
        policy.configure_channel(pi_los, 1.0 - p_ln_pre, p_nl_pre)

    los = initial_rng.random(n_uav) < pi_los
    actual_age = np.ones(n_uav)
    conservative_age = np.ones(n_uav)
    queue = np.zeros(n_uav)
    event = initial_rng.random(n_uav) < cfg.p_event
    attempted_history = []
    ack_history = []
    success_prob_history = []
    pre_age = post_age = 0.0
    pre_count = post_count = 0
    post_energy = np.zeros(n_uav)
    queue_max = np.zeros(n_uav)

    for t in range(cfg.T):
        dwell = dwell_pre if t < shift_slot else dwell_post
        p_nl = np.full(n_uav, 1.0 / dwell)
        p_ln = np.clip(p_nl * (1.0 - pi_los) / pi_los, 0.0, 1.0)
        policy.horizon = min(t, extra_delay)
        expected_age = A.expected_pending_age(
            conservative_age,
            attempted_history,
            success_prob_history,
            t,
            extra_delay,
        )
        decision_age = (
            expected_age
            if policy.age_mode == "expected"
            else conservative_age
        )
        value = np.where(event, cfg.wev, 1.0)
        selected = policy(decision_age, None, queue, value, pth, cfg)
        predicted_success = np.asarray(
            policy.success_probability(), dtype=float
        )
        energy = np.zeros(n_uav)
        attempted = np.zeros(n_uav, dtype=bool)
        ack = np.zeros(n_uav, dtype=bool)
        ack_draw = ack_rng.random(n_uav)
        for n in selected:
            energy[n] = pth[n]
            attempted[n] = True
            ack[n] = ack_draw[n] < (
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

        u = channel_rng.random(n_uav)
        los = np.where(los, u >= p_ln, u < p_nl)
        u = event_rng.random(n_uav)
        event = np.where(event, u >= cfg.ev_off, u < cfg.p_event)

        if cfg.warmup <= t < shift_slot:
            pre_age += float(np.sum(value * actual_age))
            pre_count += 1
        elif t >= shift_slot:
            post_age += float(np.sum(value * actual_age))
            post_energy += energy
            post_count += 1

    avg_power = post_energy / post_count
    shielded = bool(getattr(policy, "shielded", False))
    return {
        "policy": policy.name,
        "seed": seed,
        "extra_delay": extra_delay,
        "dwell_pre": dwell_pre,
        "dwell_post": dwell_post,
        "pre_wAoI": pre_age / pre_count,
        "post_wAoI": post_age / post_count,
        "post_max_avg_power": float(np.max(avg_power)),
        "max_queue": float(np.max(queue_max)),
        "certificate_ok": bool(
            shielded and np.max(queue_max) <= cfg.qcap + 1e-9
        ),
    }


SUMMARY_METRICS = (
    "wAoI",
    "worst_user_mean_aoi",
    "p95_user_mean_aoi",
    "p95_slot_aoi",
    "p99_slot_aoi",
    "cvar95_slot_aoi",
    "slot_aoi_gt40",
    "slot_aoi_gt60",
    "mean_service_rate",
    "worst_service_rate",
    "jain_service_fairness",
    "starvation_p95",
    "starvation_max",
    "rejection_rate",
    "mean_power",
    "max_avg_power",
    "max_queue",
)


def summarize(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    agg = {}
    for metric in SUMMARY_METRICS:
        agg[metric] = (metric, "mean")
        agg[metric + "_std"] = (metric, "std")
    agg["bound_all_seeds"] = ("bound_ok", "all")
    agg["certificate_all_seeds"] = ("certificate_ok", "all")
    agg["seeds"] = ("seed", "nunique")
    return frame.groupby(groups, as_index=False).agg(**agg)


def paired_effects(frame: pd.DataFrame) -> pd.DataFrame:
    ours = "FreshSky-Safe"
    rows = []
    for delay, part in frame.groupby("extra_delay"):
        pivot = part.pivot(index="seed", columns="policy", values="wAoI")
        for baseline in pivot.columns:
            if baseline == ours:
                continue
            valid = pivot[[ours, baseline]].dropna()
            effect = 100.0 * (valid[baseline] - valid[ours]) / valid[baseline]
            mean, ci = mean_ci95(effect)
            rows.append(
                {
                    "extra_delay": delay,
                    "baseline": baseline,
                    "freshsky_gain_pct": mean,
                    "ci95_halfwidth": ci,
                    "paired_seeds": len(valid),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run only five seeds at delay four for rapid policy screening",
    )
    parser.add_argument(
        "--zhu-only",
        action="store_true",
        help="rerun and merge only the corrected Zhu-style policy",
    )
    parser.add_argument(
        "--new-baselines-only",
        action="store_true",
        help=(
            "run Wang'26 PORMAB-Safe and Tripathi'24 Whittle-Safe over "
            "the full delay/seed grid, then merge them into existing CSVs"
        ),
    )
    parser.add_argument(
        "--zhao-only",
        action="store_true",
        help=(
            "run Zhao'25 MMSE-MW-Safe over the full delay/seed grid, "
            "then merge it into the existing baseline CSVs"
        ),
    )
    parser.add_argument(
        "--extend-evidence-only",
        action="store_true",
        help=(
            "extend the factorial and Qmax sweeps from 10 to 20 matched "
            "seeds and add exploratory Qmax points 28 and 36"
        ),
    )
    parser.add_argument(
        "--shift-only",
        action="store_true",
        help="run the matched delayed-feedback transition-shift suite",
    )
    parser.add_argument(
        "--refresh-derived",
        action="store_true",
        help="refresh summaries and paired intervals from existing seed CSV",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    B, E, D, A = load_project(args.project)
    Ours, baseline_policies, factorial = build_policy_classes(B, E, D)

    cfg = E.clone_cfg(T=4000, pbar=0.4, qcap=24.0, mean_dwell=8.0)
    if args.refresh_derived:
        seeds = pd.read_csv(
            args.output / "tmc_extended_baselines_seeds.csv"
        )
        summarize(seeds, ["policy", "extra_delay"]).to_csv(
            args.output / "tmc_extended_baselines_summary.csv",
            index=False,
        )
        paired = paired_effects(seeds)
        paired.to_csv(
            args.output / "tmc_extended_baselines_paired.csv",
            index=False,
        )
        print(paired.to_string(index=False))
        return
    if args.shift_only:
        wanted = {
            "FreshSky-Safe",
            "Static-Geometry Safe",
            "Ji'24-style Greedy-Safe",
            "Zhu'26-style Aggregate-DPP",
        }
        policies = [p for p in baseline_policies if p.name in wanted]
        seeds = pd.DataFrame(
            [
                run_shift_one(B, E, A, p, cfg, 4, seed)
                for p in policies
                for seed in range(20)
            ]
        )
        summary = (
            seeds.groupby(["policy", "extra_delay"], as_index=False)
            .agg(
                pre_wAoI=("pre_wAoI", "mean"),
                pre_wAoI_std=("pre_wAoI", "std"),
                post_wAoI=("post_wAoI", "mean"),
                post_wAoI_std=("post_wAoI", "std"),
                post_max_avg_power=("post_max_avg_power", "mean"),
                max_queue=("max_queue", "max"),
                certificate_all_seeds=("certificate_ok", "all"),
                seeds=("seed", "nunique"),
            )
        )
        seeds.to_csv(
            args.output / "tmc_delayed_shift_seeds.csv", index=False
        )
        summary.to_csv(
            args.output / "tmc_delayed_shift_summary.csv", index=False
        )
        print(summary.to_string(index=False))
        return
    if args.zhu_only:
        zhu = next(
            p
            for p in baseline_policies
            if p.name == "Zhu'26-style Aggregate-DPP"
        )
        corrected = pd.DataFrame(
            [
                run_one(B, E, A, zhu, cfg, delay, seed)
                for delay in (2, 4, 8)
                for seed in range(20)
            ]
        )
        seed_path = args.output / "tmc_extended_baselines_seeds.csv"
        previous = pd.read_csv(seed_path)
        previous = previous[
            previous["policy"] != "Zhu'26-style Aggregate-DPP"
        ]
        merged = pd.concat([previous, corrected], ignore_index=True)
        merged.to_csv(seed_path, index=False)
        summarize(merged, ["policy", "extra_delay"]).to_csv(
            args.output / "tmc_extended_baselines_summary.csv",
            index=False,
        )
        paired_effects(merged).to_csv(
            args.output / "tmc_extended_baselines_paired.csv",
            index=False,
        )
        print(
            summarize(corrected, ["policy", "extra_delay"])[
                [
                    "policy",
                    "extra_delay",
                    "wAoI",
                    "max_avg_power",
                    "max_queue",
                ]
            ].to_string(index=False)
        )
        return
    if args.zhao_only:
        wanted = {"Zhao'25 MMSE-MW-Safe"}
        policy = next(
            p for p in baseline_policies if p.name in wanted
        )
        added = pd.DataFrame(
            [
                run_one(B, E, A, policy, cfg, delay, seed)
                for delay in (2, 4, 8)
                for seed in range(20)
            ]
        )
        seed_path = args.output / "tmc_extended_baselines_seeds.csv"
        previous = pd.read_csv(seed_path)
        previous = previous[~previous["policy"].isin(wanted)]
        merged = pd.concat([previous, added], ignore_index=True)
        merged.to_csv(seed_path, index=False)
        summarize(merged, ["policy", "extra_delay"]).to_csv(
            args.output / "tmc_extended_baselines_summary.csv",
            index=False,
        )
        paired_effects(merged).to_csv(
            args.output / "tmc_extended_baselines_paired.csv",
            index=False,
        )
        print(
            summarize(added, ["policy", "extra_delay"])[
                [
                    "policy",
                    "extra_delay",
                    "wAoI",
                    "worst_user_mean_aoi",
                    "jain_service_fairness",
                    "max_avg_power",
                    "certificate_all_seeds",
                ]
            ].to_string(index=False)
        )
        return
    if args.new_baselines_only:
        wanted = {
            "Wang'26 PORMAB-Safe",
            "Tripathi'24 Whittle-Safe",
        }
        policies = [p for p in baseline_policies if p.name in wanted]
        added = pd.DataFrame(
            [
                run_one(B, E, A, policy, cfg, delay, seed)
                for policy in policies
                for delay in (2, 4, 8)
                for seed in range(20)
            ]
        )
        seed_path = args.output / "tmc_extended_baselines_seeds.csv"
        previous = pd.read_csv(seed_path)
        previous = previous[~previous["policy"].isin(wanted)]
        merged = pd.concat([previous, added], ignore_index=True)
        merged.to_csv(seed_path, index=False)
        summarize(merged, ["policy", "extra_delay"]).to_csv(
            args.output / "tmc_extended_baselines_summary.csv",
            index=False,
        )
        paired_effects(merged).to_csv(
            args.output / "tmc_extended_baselines_paired.csv",
            index=False,
        )
        print(
            summarize(added, ["policy", "extra_delay"])[
                [
                    "policy",
                    "extra_delay",
                    "wAoI",
                    "worst_user_mean_aoi",
                    "jain_service_fairness",
                    "max_avg_power",
                    "certificate_all_seeds",
                ]
            ].to_string(index=False)
        )
        return
    if args.extend_evidence_only:
        factorial_path = args.output / "tmc_factorial_ablation_seeds.csv"
        previous_factorial = pd.read_csv(factorial_path)
        factorial_rows = []
        for (adaptive, age_mode, shielded), policy in factorial.items():
            for seed in range(10, 20):
                row = run_one(B, E, A, policy, cfg, 4, seed)
                row.update(
                    adaptive_belief=adaptive,
                    age_estimator=age_mode,
                    shield=shielded,
                )
                factorial_rows.append(row)
        factorial_added = pd.DataFrame(factorial_rows)
        factorial_keys = [
            "policy",
            "extra_delay",
            "seed",
        ]
        previous_factorial = previous_factorial.merge(
            factorial_added[factorial_keys],
            on=factorial_keys,
            how="left",
            indicator=True,
        )
        previous_factorial = previous_factorial[
            previous_factorial["_merge"] == "left_only"
        ].drop(columns="_merge")
        factorial_seeds = pd.concat(
            [previous_factorial, factorial_added], ignore_index=True
        )
        factorial_seeds.to_csv(factorial_path, index=False)
        summarize(
            factorial_seeds,
            ["adaptive_belief", "age_estimator", "shield", "policy"],
        ).to_csv(
            args.output / "tmc_factorial_ablation_summary.csv",
            index=False,
        )

        qcap_path = args.output / "tmc_qcap_service_seeds.csv"
        previous_qcap = pd.read_csv(qcap_path)
        qcap_rows = []
        for qcap in (4.0, 8.0, 12.0, 16.0, 24.0, 28.0, 32.0, 36.0, 48.0):
            cap_cfg = E.clone_cfg(
                T=4000, pbar=0.4, qcap=qcap, mean_dwell=8.0
            )
            start_seed = 0 if qcap in (28.0, 36.0) else 10
            for seed in range(start_seed, 20):
                row = run_one(B, E, A, Ours, cap_cfg, 4, seed)
                row["qcap"] = qcap
                qcap_rows.append(row)
        qcap_added = pd.DataFrame(qcap_rows)
        qcap_keys = ["qcap", "seed"]
        previous_qcap = previous_qcap.merge(
            qcap_added[qcap_keys],
            on=qcap_keys,
            how="left",
            indicator=True,
        )
        previous_qcap = previous_qcap[
            previous_qcap["_merge"] == "left_only"
        ].drop(columns="_merge")
        qcap_seeds = pd.concat(
            [previous_qcap, qcap_added], ignore_index=True
        )
        qcap_seeds.to_csv(qcap_path, index=False)
        summarize(qcap_seeds, ["qcap"]).to_csv(
            args.output / "tmc_qcap_service_summary.csv",
            index=False,
        )
        print(
            "extended factorial seeds per cell:",
            factorial_seeds.groupby(
                ["adaptive_belief", "age_estimator", "shield"]
            )["seed"].nunique().min(),
        )
        print(
            "extended Qmax seeds per point:",
            qcap_seeds.groupby("qcap")["seed"].nunique().to_dict(),
        )
        return
    if args.quick:
        quick_rows = [
            run_one(B, E, A, policy, cfg, 4, seed)
            for policy in baseline_policies
            for seed in range(5)
        ]
        quick_seeds = pd.DataFrame(quick_rows)
        quick_summary = summarize(
            quick_seeds, ["policy", "extra_delay"]
        )
        quick_seeds.to_csv(
            args.output / "tmc_quick_screen_seeds.csv", index=False
        )
        quick_summary.to_csv(
            args.output / "tmc_quick_screen_summary.csv", index=False
        )
        print(
            quick_summary[
                [
                    "policy",
                    "wAoI",
                    "worst_user_mean_aoi",
                    "rejection_rate",
                    "max_avg_power",
                    "certificate_all_seeds",
                ]
            ].to_string(index=False)
        )
        return
    baseline_rows = [
        run_one(B, E, A, policy, cfg, delay, seed)
        for policy in baseline_policies
        for delay in (2, 4, 8)
        for seed in range(20)
    ]
    baseline_seeds = pd.DataFrame(baseline_rows)
    baseline_summary = summarize(
        baseline_seeds, ["policy", "extra_delay"]
    )
    baseline_paired = paired_effects(baseline_seeds)

    factorial_rows = []
    for (adaptive, age_mode, shielded), policy in factorial.items():
        for seed in range(20):
            row = run_one(B, E, A, policy, cfg, 4, seed)
            row.update(
                adaptive_belief=adaptive,
                age_estimator=age_mode,
                shield=shielded,
            )
            factorial_rows.append(row)
    factorial_seeds = pd.DataFrame(factorial_rows)
    factorial_summary = summarize(
        factorial_seeds,
        ["adaptive_belief", "age_estimator", "shield", "policy"],
    )

    qcap_rows = []
    for qcap in (
        4.0,
        8.0,
        12.0,
        16.0,
        24.0,
        28.0,
        32.0,
        36.0,
        48.0,
    ):
        cap_cfg = E.clone_cfg(
            T=4000, pbar=0.4, qcap=qcap, mean_dwell=8.0
        )
        for seed in range(20):
            row = run_one(B, E, A, Ours, cap_cfg, 4, seed)
            row["qcap"] = qcap
            qcap_rows.append(row)
    qcap_seeds = pd.DataFrame(qcap_rows)
    qcap_summary = summarize(qcap_seeds, ["qcap"])

    outputs = {
        "tmc_extended_baselines_seeds.csv": baseline_seeds,
        "tmc_extended_baselines_summary.csv": baseline_summary,
        "tmc_extended_baselines_paired.csv": baseline_paired,
        "tmc_factorial_ablation_seeds.csv": factorial_seeds,
        "tmc_factorial_ablation_summary.csv": factorial_summary,
        "tmc_qcap_service_seeds.csv": qcap_seeds,
        "tmc_qcap_service_summary.csv": qcap_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output / name, index=False)

    print("\n=== Expanded delayed-feedback baselines ===")
    show = [
        "policy",
        "extra_delay",
        "wAoI",
        "worst_user_mean_aoi",
        "jain_service_fairness",
        "starvation_p95",
        "rejection_rate",
        "max_avg_power",
        "certificate_all_seeds",
    ]
    print(baseline_summary[show].to_string(index=False))
    print("\n=== 2x2x2 mechanism ablation (d=4) ===")
    show = [
        "adaptive_belief",
        "age_estimator",
        "shield",
        "wAoI",
        "worst_user_mean_aoi",
        "rejection_rate",
        "max_avg_power",
        "certificate_all_seeds",
    ]
    print(factorial_summary[show].to_string(index=False))
    print("\n=== Qmax service frontier (d=4) ===")
    show = [
        "qcap",
        "wAoI",
        "worst_user_mean_aoi",
        "p95_slot_aoi",
        "jain_service_fairness",
        "starvation_p95",
        "rejection_rate",
        "max_avg_power",
    ]
    print(qcap_summary[show].to_string(index=False))
    print("\nSaved:")
    for name in outputs:
        print(args.output / name)


if __name__ == "__main__":
    main()
