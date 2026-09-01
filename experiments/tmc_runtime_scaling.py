# -*- coding: utf-8 -*-
"""Runtime and memory scaling for delayed Bayesian FreshSky."""
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd

import belief_whittle as B
import bayes_hmm_experiment as E
import bayes_hmm_delayed_aoi_belief as A
import bayes_hmm_grid_sensitivity as G


N_VALUES = (12, 25, 50, 100, 200, 500)
GRID_SIZES = (6, 12, 20)
DELAYS = (0, 4, 8)
REPEATS = 1000


def policy_class(grid_size):
    class BenchPolicy(A.ExpectedAoIBayesSafe):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.filter = G.GridFilter(cfg.N, grid_size)

        def __call__(self, age, unused, queue, value, pth, cfg):
            del unused
            score = (
                cfg.V * value * age * self.success_probability()
                - queue * pth
            )
            next_q = np.maximum(queue - cfg.pbar, 0.0) + pth
            score = np.where(
                next_q <= self.qcap + 1e-12, score, -1e18
            )
            return B.topM(score, cfg)

    return BenchPolicy


def benchmark(n_uav, grid_size, delay):
    cfg = E.clone_cfg(
        N=n_uav,
        M=max(1, n_uav // 4),
        pbar=0.4,
        qcap=24.0,
        mean_dwell=8.0,
    )
    policy = policy_class(grid_size)(cfg)
    policy.horizon = delay
    rng = np.random.default_rng(
        880000 + 1000 * n_uav + 10 * grid_size + delay
    )
    age = rng.uniform(1.0, 60.0, n_uav)
    queue = rng.uniform(0.0, 20.0, n_uav)
    value = rng.choice((1.0, cfg.wev), n_uav)
    pth = rng.uniform(0.1, 2.0, n_uav)
    attempted_history = [
        rng.random(n_uav) < 0.25 for _ in range(delay)
    ]
    success_history = [
        rng.uniform(0.05, 0.95, n_uav) for _ in range(delay)
    ]
    attempted = rng.random(n_uav) < 0.25
    ack = attempted & (rng.random(n_uav) < 0.65)

    for _ in range(20):
        policy(age, None, queue, value, pth, cfg)
        policy.observe_feedback(attempted, ack)

    expected_ns, decision_ns, accounting_ns, update_ns = [], [], [], []
    for repeat in range(REPEATS):
        t0 = time.perf_counter_ns()
        decision_age = A.expected_pending_age(
            age,
            attempted_history,
            success_history,
            delay,
            delay,
        )
        t1 = time.perf_counter_ns()
        selected = policy(
            decision_age, None, queue, value, pth, cfg
        )
        t2 = time.perf_counter_ns()
        energy = np.zeros(n_uav)
        selected_array = np.asarray(selected, dtype=int)
        energy[selected_array] = pth[selected_array]
        queue = np.maximum(queue - cfg.pbar, 0.0) + energy
        if delay:
            history_slot = repeat % delay
            attempted_history[history_slot][...] = attempted
            success_history[history_slot][...] = (
                policy.success_probability()
            )
        t3 = time.perf_counter_ns()
        policy.observe_feedback(attempted, ack)
        t4 = time.perf_counter_ns()
        expected_ns.append(t1 - t0)
        decision_ns.append(t2 - t1)
        accounting_ns.append(t3 - t2)
        update_ns.append(t4 - t3)

    arrays = (
        policy.filter.pLL,
        policy.filter.pNL,
        policy.filter.weight,
        policy.filter.theta,
        age,
        queue,
        value,
        pth,
    )
    state_bytes = int(sum(x.nbytes for x in arrays))
    history_bytes = int(
        sum(x.nbytes for x in attempted_history)
        + sum(x.nbytes for x in success_history)
    )
    total_us = (
        np.asarray(expected_ns)
        + np.asarray(decision_ns)
        + np.asarray(accounting_ns)
        + np.asarray(update_ns)
    ) / 1000.0
    return dict(
        N=n_uav,
        M=cfg.M,
        grid_size=grid_size,
        retained_models=policy.filter.K,
        extra_delay=delay,
        repeats=REPEATS,
        expected_age_median_us=np.median(expected_ns) / 1000.0,
        decision_median_us=np.median(decision_ns) / 1000.0,
        accounting_median_us=np.median(accounting_ns) / 1000.0,
        filter_update_median_us=np.median(update_ns) / 1000.0,
        total_median_us=float(np.median(total_us)),
        total_p95_us=float(np.percentile(total_us, 95)),
        state_memory_kib=state_bytes / 1024.0,
        delay_history_kib=history_bytes / 1024.0,
        total_memory_kib=(state_bytes + history_bytes) / 1024.0,
        within_100ms_p95=bool(np.percentile(total_us, 95) < 100000.0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(B.RES),
        help="directory for tmc_runtime_scaling.csv",
    )
    args = parser.parse_args()
    rows = [
        benchmark(n_uav, grid_size, delay)
        for n_uav in N_VALUES
        for grid_size in GRID_SIZES
        for delay in DELAYS
    ]
    result = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "tmc_runtime_scaling.csv"
    result.to_csv(path, index=False)
    print(result.to_string(index=False))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
