# -*- coding: utf-8 -*-
"""Model-mismatch evaluation on cached non-Markov OSM ray-traced traces."""
import os
import numpy as np
import pandas as pd
import belief_whittle as B
import bayes_hmm_experiment as E
import urban_blockage as U

TRACES = {
    "Manhattan": "osm_traces_midtown_manhattan.npz",
    "London": "osm_traces_city_of_london.npz",
}

POLICIES = [
    ("Bayes-FreshSky-Safe", E.BayesFreshSkySafe),
    ("Bayes-FreshSky", E.BayesFreshSky),
    ("Bayes-Aggregate-DPP", E.BayesAggregateDPP),
    ("TS-DPP adapted", B.TSDPPAdapted26),
    ("FreshSky-B fixed-GE", B.belief_whittle),
    ("Wang'26 fixed-GE", B.wang26),
    ("Memoryless", B.memoryless),
]


def run_trace(policy, cfg, los, dist, seed):
    rng = np.random.default_rng(29000 + seed)
    n_uav = cfg.N
    pol = policy(cfg) if isinstance(policy, type) else policy
    if hasattr(pol, "reset_seed"):
        pol.reset_seed(seed)
    fspl = 20 * np.log10(4 * np.pi * U.FC * dist / 3e8)
    pth = np.minimum(
        U.GAMMA * U.SIGMA2 / 10 ** (-(fspl + U.ETAL) / 10), U.PMAX
    )
    pi_los = np.clip(los.mean(axis=0), 0.05, 0.95)
    p_nl = 1.0 / 8.0
    p_ln = np.clip(p_nl * (1.0 - pi_los) / pi_los, 0.0, 1.0)
    B._ALPHA = 1.0 - p_ln
    B._BETA = 1.0 - p_nl
    fixed_belief = pi_los.copy()
    age = np.ones(n_uav)
    queue = np.zeros(n_uav)
    event = rng.random(n_uav) < cfg.p_event
    weighted_age = 0.0
    count = 0
    energy_sum = np.zeros(n_uav)
    queue_max = np.zeros(n_uav)
    for t in range(cfg.T):
        value = np.where(event, cfg.wev, 1.0)
        selected = pol(age, fixed_belief, queue, value, pth[t], cfg)
        energy = np.zeros(n_uav)
        ack = np.zeros(n_uav, dtype=bool)
        attempted = np.zeros(n_uav, dtype=bool)
        for n in selected:
            energy[n] = pth[t, n]
            attempted[n] = True
            ack[n] = rng.random() < (
                B.PHI_LOS if los[t, n] else B.PHI_NLOS
            )
        if hasattr(pol, "update"):
            pol.update(energy)
        if hasattr(pol, "observe_feedback"):
            pol.observe_feedback(attempted, ack)
        age = np.where(ack, 1.0, np.minimum(age + 1.0, cfg.Amax))
        queue = np.maximum(queue - cfg.pbar, 0.0) + energy
        queue_max = np.maximum(queue_max, queue)
        p_ack = fixed_belief * B.PHI_LOS + (1.0 - fixed_belief) * B.PHI_NLOS
        posterior = fixed_belief.copy()
        yes = attempted & ack
        no = attempted & ~ack
        posterior[yes] = (
            fixed_belief[yes] * B.PHI_LOS / (p_ack[yes] + 1e-12)
        )
        posterior[no] = (
            fixed_belief[no] * (1.0 - B.PHI_LOS)
            / (1.0 - p_ack[no] + 1e-12)
        )
        fixed_belief = posterior * (1.0 - p_ln) + (1.0 - posterior) * p_nl
        u = rng.random(n_uav)
        event = np.where(event, u >= cfg.ev_off, u < cfg.p_event)
        if t >= cfg.warmup:
            weighted_age += float(np.sum(value * age))
            energy_sum += energy
            count += 1
    return dict(
        seed=seed,
        wAoI=weighted_age / count,
        mean_power=float(np.mean(energy_sum / count)),
        max_avg_power=float(np.max(energy_sum / count)),
        max_queue=float(np.max(queue_max)),
        feasible=bool(np.all(energy_sum / count <= cfg.pbar * 1.02)),
        certificate_ok=bool(
            not hasattr(pol, "qcap")
            or np.max(queue_max) <= pol.qcap + 1e-9
        ),
    )


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    cfg = E.clone_cfg(T=2000, warmup=300, pbar=0.4, qcap=24.0)
    all_seeds = []
    summaries = []
    for city, filename in TRACES.items():
        trace = np.load(os.path.join(data_dir, filename))
        los, dist = trace["los"], trace["dist"]
        print(f"== {city}: LoS={los.mean():.3f}, T={len(los)} ==")
        for name, policy in POLICIES:
            rows = [run_trace(policy, cfg, los, dist, seed) for seed in range(5)]
            frame = pd.DataFrame(rows)
            frame.insert(0, "policy", name)
            frame.insert(0, "city", city)
            all_seeds.append(frame)
            summary = dict(
                city=city,
                policy=name,
                wAoI=frame.wAoI.mean(),
                wAoI_std=frame.wAoI.std(ddof=1),
                mean_power=frame.mean_power.mean(),
                max_avg_power=frame.max_avg_power.mean(),
                max_queue=frame.max_queue.max(),
                feasible_all_seeds=frame.feasible.all(),
                certificate_all_seeds=frame.certificate_ok.all(),
            )
            summaries.append(summary)
            print(
                f"{name:23s} W-AoI={summary['wAoI']:7.1f}"
                f"+/-{summary['wAoI_std']:5.1f} maxP={summary['max_avg_power']:.3f} "
                f"maxQ={summary['max_queue']:.1f} feasible={summary['feasible_all_seeds']}"
            )
    os.makedirs(B.RES, exist_ok=True)
    pd.DataFrame(summaries).to_csv(
        os.path.join(B.RES, "bayes_hmm_osm_summary.csv"), index=False
    )
    pd.concat(all_seeds, ignore_index=True).to_csv(
        os.path.join(B.RES, "bayes_hmm_osm_seeds.csv"), index=False
    )
    print("saved bayes_hmm_osm_summary.csv and bayes_hmm_osm_seeds.csv")


if __name__ == "__main__":
    main()
