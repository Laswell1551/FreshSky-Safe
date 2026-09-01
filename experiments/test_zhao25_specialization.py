"""Formula-level fidelity checks for the Zhao--Kadota'25 baseline."""
from __future__ import annotations

import numpy as np

import bayes_hmm_delayed_aoi_belief as delayed_age
import bayes_hmm_experiment as experiment
import bayes_hmm_delayed_ack as delayed_ack
import belief_whittle as belief
from tmc_extended_evidence import (
    build_policy_classes,
    zhao_mmse_maxweight_priority,
    zhao_optimal_randomized_rates,
)


def test_pending_age_specialization() -> None:
    conservative = np.array([7.0])

    no_pending = delayed_age.expected_pending_age(
        conservative, [], [], t=0, delay=0
    )
    np.testing.assert_allclose(no_pending, [7.0])

    one_pending = delayed_age.expected_pending_age(
        conservative,
        [np.array([True])],
        [np.array([0.4])],
        t=1,
        delay=1,
    )
    np.testing.assert_allclose(one_pending, [4.6])

    two_pending = delayed_age.expected_pending_age(
        conservative,
        [np.array([True]), np.array([True])],
        [np.array([0.4]), np.array([0.5])],
        t=2,
        delay=2,
    )
    np.testing.assert_allclose(two_pending, [3.0])


def test_score_specialization_and_nonduplication() -> None:
    value = np.array([1.0, 4.0, 2.0])
    success = np.array([0.3, 0.7, 0.5])
    age = np.array([8.0, 2.0, 5.0])
    zeros = np.zeros(3)
    ones = np.ones(3)

    specialized = zhao_mmse_maxweight_priority(
        value, success, ones, age, zeros, zeros
    )
    ji_score = value * success * age
    np.testing.assert_allclose(specialized, ji_score)

    rates = zhao_optimal_randomized_rates(
        np.ones(3), success, capacity=2.0
    )
    beta = 1.0 / (success * rates)
    theorem_score = zhao_mmse_maxweight_priority(
        beta, success, ones, age, zeros, zeros
    )
    assert not np.allclose(theorem_score, ji_score)


def test_theorem_beta_water_filling() -> None:
    alpha = np.array([1.0, 2.0, 0.5, 3.0])
    success = np.array([0.2, 0.4, 0.8, 0.6])
    rates = zhao_optimal_randomized_rates(alpha, success, capacity=2.5)
    assert np.all(rates > 0.0)
    assert np.all(rates <= 1.0)
    np.testing.assert_allclose(np.sum(rates), 2.5, atol=1e-10)
    beta = alpha / (success * rates)
    np.testing.assert_allclose(beta * success * rates, alpha)


def test_common_shield_rejects_infeasible_top_score() -> None:
    _, policies, _ = build_policy_classes(belief, experiment, delayed_ack)
    zhao = next(
        p for p in policies if p.name == "Zhao'25 MMSE-MW-Safe"
    )
    cfg = experiment.clone_cfg(T=10, pbar=0.4, qcap=1.0)
    policy = zhao(cfg)
    pi_los = np.linspace(0.35, 0.75, cfg.N)
    p_nl = np.full(cfg.N, 1.0 / cfg.mean_dwell)
    p_ln = np.clip(p_nl * (1.0 - pi_los) / pi_los, 0.0, 1.0)
    policy.configure_channel(pi_los, 1.0 - p_ln, p_nl)

    policy.beta = np.ones(cfg.N)
    age = np.arange(cfg.N, 0, -1, dtype=float)
    queue = np.zeros(cfg.N)
    pth = np.full(cfg.N, 0.2)
    pth[0] = 0.8
    queue[0] = cfg.qcap
    selected = policy(age, None, queue, np.ones(cfg.N), pth, cfg)
    assert 0 not in selected
    assert len(selected) == cfg.M
    next_queue = np.maximum(queue - cfg.pbar, 0.0) + pth
    assert np.all(next_queue[selected] <= cfg.qcap + 1e-12)


def main() -> None:
    test_pending_age_specialization()
    test_score_specialization_and_nonduplication()
    test_theorem_beta_water_filling()
    test_common_shield_rejects_infeasible_top_score()
    print("PASS: Zhao'25 specialization, calibration, and shield checks")


if __name__ == "__main__":
    main()
