# -*- coding: utf-8 -*-
"""Abrupt hidden-channel transition shift for learned and oracle baselines."""
import os
import numpy as np
import pandas as pd
import belief_whittle as B
import bayes_hmm_experiment as E

T_SHIFT = 2000
DWELL_PRE = 2.0
DWELL_POST = 14.0
SEEDS = (0, 1, 2, 3, 4)

POLICIES = [
    ("Bayes-FreshSky-Safe", E.BayesFreshSkySafe),
    ("Bayes-FreshSky", E.BayesFreshSky),
    ("Bayes-Aggregate-DPP", E.BayesAggregateDPP),
    ("TS-DPP adapted", B.TSDPPAdapted26),
    ("FreshSky-B oracle", B.belief_whittle),
    ("Wang'26 oracle", B.wang26),
    ("Memoryless", B.memoryless),
]


def run_one(policy, cfg, seed):
    rng = np.random.default_rng(17000 + seed)
    n_uav = cfg.N
    pol = policy(cfg) if isinstance(policy, type) else policy
    if hasattr(pol, "reset_seed"):
        pol.reset_seed(seed)
    pi_los, pth = B.geometry(cfg, rng)
    los = rng.random(n_uav) < pi_los
    oracle_belief = pi_los.copy()
    age = np.ones(n_uav)
    queue = np.zeros(n_uav)
    event = rng.random(n_uav) < cfg.p_event
    pre_age = post_age = 0.0
    pre_count = post_count = 0
    post_energy = np.zeros(n_uav)
    queue_max = np.zeros(n_uav)
    for t in range(cfg.T):
        dwell = DWELL_PRE if t < T_SHIFT else DWELL_POST
        p_nl = 1.0 / dwell
        p_ln = np.clip(p_nl * (1.0 - pi_los) / pi_los, 0.0, 1.0)
        B._ALPHA = 1.0 - p_ln
        B._BETA = 1.0 - p_nl
        value = np.where(event, cfg.wev, 1.0)
        selected = pol(age, oracle_belief, queue, value, pth, cfg)
        energy = np.zeros(n_uav)
        ack = np.zeros(n_uav, dtype=bool)
        attempted = np.zeros(n_uav, dtype=bool)
        for n in selected:
            energy[n] = pth[n]
            attempted[n] = True
            ack[n] = rng.random() < (B.PHI_LOS if los[n] else B.PHI_NLOS)
        if hasattr(pol, "update"):
            pol.update(energy)
        if hasattr(pol, "observe_feedback"):
            pol.observe_feedback(attempted, ack)
        age = np.where(ack, 1.0, age + 1.0)
        queue = np.maximum(queue - cfg.pbar, 0.0) + energy
        queue_max = np.maximum(queue_max, queue)
        p_ack = oracle_belief * B.PHI_LOS + (1.0 - oracle_belief) * B.PHI_NLOS
        post = oracle_belief.copy()
        yes = attempted & ack
        no = attempted & ~ack
        post[yes] = oracle_belief[yes] * B.PHI_LOS / (p_ack[yes] + 1e-12)
        post[no] = (
            oracle_belief[no] * (1.0 - B.PHI_LOS)
            / (1.0 - p_ack[no] + 1e-12)
        )
        oracle_belief = post * (1.0 - p_ln) + (1.0 - post) * p_nl
        u = rng.random(n_uav)
        los = np.where(los, u >= p_ln, u < p_nl)
        u = rng.random(n_uav)
        event = np.where(event, u >= cfg.ev_off, u < cfg.p_event)
        if cfg.warmup <= t < T_SHIFT:
            pre_age += float(np.sum(value * age))
            pre_count += 1
        elif t >= T_SHIFT:
            post_age += float(np.sum(value * age))
            post_energy += energy
            post_count += 1
    entropy = (
        float(np.mean(pol.filter.normalized_entropy))
        if hasattr(pol, "filter") else np.nan
    )
    return dict(
        seed=seed,
        pre_wAoI=pre_age / pre_count,
        post_wAoI=post_age / post_count,
        post_mean_power=float(np.mean(post_energy / post_count)),
        post_max_avg_power=float(np.max(post_energy / post_count)),
        max_queue=float(np.max(queue_max)),
        final_model_entropy=entropy,
        post_feasible=bool(np.all(post_energy / post_count <= cfg.pbar * 1.02)),
        certificate_ok=bool(
            not hasattr(pol, "qcap")
            or np.max(queue_max) <= pol.qcap + 1e-9
        ),
    )


def main():
    cfg = E.clone_cfg(T=4000, pbar=0.4, qcap=24.0)
    seed_rows = []
    summaries = []
    for name, policy in POLICIES:
        rows = [run_one(policy, cfg, seed) for seed in SEEDS]
        frame = pd.DataFrame(rows)
        frame.insert(0, "policy", name)
        seed_rows.append(frame)
        summary = dict(
            policy=name,
            pre_wAoI=frame.pre_wAoI.mean(),
            post_wAoI=frame.post_wAoI.mean(),
            post_wAoI_std=frame.post_wAoI.std(ddof=1),
            post_max_avg_power=frame.post_max_avg_power.mean(),
            max_queue=frame.max_queue.max(),
            final_model_entropy=frame.final_model_entropy.mean(),
            post_feasible_all_seeds=frame.post_feasible.all(),
            certificate_all_seeds=frame.certificate_ok.all(),
        )
        summaries.append(summary)
        print(
            f"{name:23s} pre={summary['pre_wAoI']:7.1f} "
            f"post={summary['post_wAoI']:7.1f}+/-{summary['post_wAoI_std']:5.1f} "
            f"maxP={summary['post_max_avg_power']:.3f} maxQ={summary['max_queue']:.1f} "
            f"feasible={summary['post_feasible_all_seeds']}"
        )
    os.makedirs(B.RES, exist_ok=True)
    pd.DataFrame(summaries).to_csv(
        os.path.join(B.RES, "bayes_hmm_shift_summary.csv"), index=False
    )
    pd.concat(seed_rows, ignore_index=True).to_csv(
        os.path.join(B.RES, "bayes_hmm_shift_seeds.csv"), index=False
    )
    print("saved bayes_hmm_shift_summary.csv and bayes_hmm_shift_seeds.csv")


if __name__ == "__main__":
    main()
