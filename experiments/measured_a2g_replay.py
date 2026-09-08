# -*- coding: utf-8 -*-
"""Measurement-driven replay on an urban 3.6-GHz air-to-ground CIR trace.

Source:
  Q. Zhu et al., "Measured and RT-based A2G Channel Dataset (CIR) under
  Urban Scenarios," Mendeley Data, V2, 2025, CC BY 4.0.
  https://doi.org/10.17632/mgdjk8n9k8.2

The measured trajectory contains 1051 time-ordered CIR snapshots.  Each
snapshot is reduced to total received power by summing tap powers in the
linear domain.  Three pre-registered quantile thresholds (25/50/75%) produce
binary measured good/bad sequences.  ACKs retain the simulator's
conditional decoding probabilities.  UAVs receive phase-shifted copies of the
same measured trajectory; therefore this is trace-driven replay, not HIL and
not a multi-UAV measurement campaign.
"""
import os
import numpy as np
import pandas as pd

import belief_whittle as B
import bayes_hmm_experiment as E


DATA = os.path.join(
    os.path.dirname(__file__), "data", "measured_a2g_urban_3p6ghz.xlsx"
)
SEEDS = (0, 1, 2, 3, 4)
QUANTILES = (0.25, 0.50, 0.75)

POLICIES = (
    ("Bayes-FreshSky-Safe", E.BayesFreshSkySafe),
    ("Bayes-FreshSky", E.BayesFreshSky),
    ("Bayes-Aggregate-DPP", E.BayesAggregateDPP),
    ("TS-DPP adapted", B.TSDPPAdapted26),
    ("FreshSky-B fitted-GE", B.belief_whittle),
    ("Wang'26 fitted-GE", B.wang26),
    ("Memoryless", B.memoryless),
)


def load_received_power():
    raw = pd.read_excel(DATA, header=None)
    taps_db = (
        raw.iloc[3:, 4:].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    )
    linear = np.nansum(np.power(10.0, taps_db / 10.0), axis=1)
    if np.any(~np.isfinite(linear)) or np.any(linear <= 0):
        raise ValueError("invalid measured CIR power")
    return 10.0 * np.log10(linear)


def fit_binary_ge(good):
    current = good[:-1]
    nxt = good[1:]
    p_ll = np.mean(nxt[current]) if np.any(current) else 0.5
    p_nl = np.mean(nxt[~current]) if np.any(~current) else 0.5
    return float(p_ll), float(p_nl)


def run_one(policy_spec, cfg, good, p_ll, p_nl, seed):
    rng = np.random.default_rng(31000 + seed)
    policy = policy_spec(cfg) if isinstance(policy_spec, type) else policy_spec
    if hasattr(policy, "reset_seed"):
        policy.reset_seed(seed)

    pi_los, pth = B.geometry(cfg, rng)
    del pi_los
    length = len(good)
    offsets = rng.integers(0, length, size=cfg.N)
    belief = np.full(cfg.N, np.mean(good), dtype=float)
    age = np.ones(cfg.N)
    queue = np.zeros(cfg.N)
    event = rng.random(cfg.N) < cfg.p_event
    B._ALPHA = p_ll
    B._BETA = 1.0 - p_nl

    weighted_age = 0.0
    energy_sum = np.zeros(cfg.N)
    queue_max = np.zeros(cfg.N)
    count = 0
    for t in range(cfg.T):
        value = np.where(event, cfg.wev, 1.0)
        selected = policy(age, belief, queue, value, pth, cfg)
        energy = np.zeros(cfg.N)
        attempted = np.zeros(cfg.N, dtype=bool)
        ack = np.zeros(cfg.N, dtype=bool)
        state = good[(offsets + t) % length]
        for n in selected:
            energy[n] = pth[n]
            attempted[n] = True
            ack[n] = rng.random() < (
                B.PHI_LOS if state[n] else B.PHI_NLOS
            )
        if hasattr(policy, "update"):
            policy.update(energy)
        if hasattr(policy, "observe_feedback"):
            policy.observe_feedback(attempted, ack)

        age = np.where(ack, 1.0, age + 1.0)
        queue = np.maximum(queue - cfg.pbar, 0.0) + energy
        queue_max = np.maximum(queue_max, queue)

        p_ack = belief * B.PHI_LOS + (1.0 - belief) * B.PHI_NLOS
        posterior = belief.copy()
        yes = attempted & ack
        no = attempted & ~ack
        posterior[yes] = belief[yes] * B.PHI_LOS / (p_ack[yes] + 1e-12)
        posterior[no] = (
            belief[no] * (1.0 - B.PHI_LOS)
            / (1.0 - p_ack[no] + 1e-12)
        )
        belief = posterior * p_ll + (1.0 - posterior) * p_nl

        u = rng.random(cfg.N)
        event = np.where(event, u >= cfg.ev_off, u < cfg.p_event)
        if t >= cfg.warmup:
            weighted_age += float(np.sum(value * age))
            energy_sum += energy
            count += 1

    avg_power = energy_sum / count
    qcap = float(getattr(policy, "qcap", np.inf))
    finite_bound = cfg.pbar + qcap / count if np.isfinite(qcap) else np.nan
    return dict(
        seed=seed,
        wAoI=weighted_age / count,
        mean_power=float(np.mean(avg_power)),
        max_avg_power=float(np.max(avg_power)),
        max_queue=float(np.max(queue_max)),
        finite_window_bound=finite_bound,
        bound_ok=(
            bool(np.max(avg_power) <= finite_bound + 1e-9)
            if np.isfinite(qcap) else None
        ),
        certificate_ok=(
            bool(np.max(queue_max) <= qcap + 1e-9)
            if np.isfinite(qcap) else None
        ),
    )


def main():
    power_db = load_received_power()
    cfg = E.clone_cfg(T=5000, warmup=500, pbar=0.4, qcap=24.0)
    seed_frames = []
    summaries = []
    for quantile in QUANTILES:
        threshold = float(np.quantile(power_db, quantile))
        good = power_db >= threshold
        p_ll, p_nl = fit_binary_ge(good)
        for name, policy in POLICIES:
            frame = pd.DataFrame(
                [
                    run_one(policy, cfg, good, p_ll, p_nl, seed)
                    for seed in SEEDS
                ]
            )
            frame.insert(0, "policy", name)
            frame.insert(0, "threshold_db", threshold)
            frame.insert(0, "good_fraction", float(np.mean(good)))
            frame.insert(0, "quantile", quantile)
            frame["fitted_pLL"] = p_ll
            frame["fitted_pNL"] = p_nl
            seed_frames.append(frame)
            summaries.append(
                dict(
                    quantile=quantile,
                    threshold_db=threshold,
                    good_fraction=float(np.mean(good)),
                    fitted_pLL=p_ll,
                    fitted_pNL=p_nl,
                    policy=name,
                    wAoI=frame.wAoI.mean(),
                    wAoI_std=frame.wAoI.std(ddof=1),
                    max_avg_power=frame.max_avg_power.mean(),
                    max_queue=frame.max_queue.max(),
                    bound_all_seeds=(
                        bool(frame.bound_ok.fillna(False).all())
                        if frame.bound_ok.notna().any() else np.nan
                    ),
                    certificate_all_seeds=(
                        bool(frame.certificate_ok.fillna(False).all())
                        if frame.certificate_ok.notna().any() else np.nan
                    ),
                )
            )

    os.makedirs(B.RES, exist_ok=True)
    summary_path = os.path.join(B.RES, "measured_a2g_replay_summary.csv")
    seed_path = os.path.join(B.RES, "measured_a2g_replay_seeds.csv")
    result = pd.DataFrame(summaries)
    result.to_csv(summary_path, index=False)
    pd.concat(seed_frames, ignore_index=True).to_csv(seed_path, index=False)
    print(result.to_string(index=False))
    print(f"saved {summary_path} and {seed_path}")


if __name__ == "__main__":
    main()
