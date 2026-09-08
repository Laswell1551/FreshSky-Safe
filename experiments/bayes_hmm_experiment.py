# -*- coding: utf-8 -*-
"""Bayesian hidden-channel learning and finite-window safety experiments.

The learner maintains a finite Bayesian model ensemble for every UAV's
Gilbert--Elliott channel.  It updates model weights and state beliefs using
only noisy ACK/NACK observations.  No true transition probability is exposed
to learned policies.

The optional safety shield accepts a transmission only when the exact next
individual virtual queue remains below qcap:

    Q_n(t+1) = max(Q_n(t)-pbar, 0) + p_n(t) <= qcap.

Since Q_n(t+1) >= Q_n(t) + p_n(t) - pbar, telescoping gives, pathwise, for
every interval [a,b],

    sum_{t=a}^b (p_n(t)-pbar) <= Q_n(b+1)-Q_n(a) <= qcap.

Thus qcap is an explicit finite-window cumulative-violation certificate.
"""
import os
import numpy as np
import pandas as pd
import belief_whittle as B


class BayesianGEFilter:
    """Finite-grid Bayesian filter for hidden two-state Markov channels."""

    def __init__(self, n_uav, forget=2e-3):
        pll = np.linspace(0.55, 0.99, 12)
        pnl = np.geomspace(0.005, 0.45, 12)
        self.pLL, self.pNL = np.meshgrid(pll, pnl, indexing="ij")
        self.pLL = self.pLL.ravel()
        self.pNL = self.pNL.ravel()
        keep = self.pLL > self.pNL
        self.pLL, self.pNL = self.pLL[keep], self.pNL[keep]
        self.K = len(self.pLL)
        self.N = n_uav
        self.forget = float(forget)
        self.weight = np.full((self.N, self.K), 1.0 / self.K)
        self.theta = np.full((self.N, self.K), 0.5)

    @property
    def belief(self):
        return np.sum(self.weight * self.theta, axis=1)

    @property
    def normalized_entropy(self):
        h = -np.sum(self.weight * np.log(self.weight + 1e-15), axis=1)
        return h / np.log(self.K)

    def step(self, attempted, ack):
        attempted = np.asarray(attempted, dtype=bool)
        ack = np.asarray(ack, dtype=bool)
        if self.forget > 0:
            self.weight = (
                (1.0 - self.forget) * self.weight + self.forget / self.K
            )
        post = self.theta.copy()
        for n in np.flatnonzero(attempted):
            p_ack = (
                self.theta[n] * B.PHI_LOS
                + (1.0 - self.theta[n]) * B.PHI_NLOS
            )
            likelihood = p_ack if ack[n] else 1.0 - p_ack
            self.weight[n] *= np.maximum(likelihood, 1e-12)
            self.weight[n] /= np.sum(self.weight[n])
            if ack[n]:
                post[n] = self.theta[n] * B.PHI_LOS / (p_ack + 1e-12)
            else:
                post[n] = (
                    self.theta[n] * (1.0 - B.PHI_LOS)
                    / (1.0 - p_ack + 1e-12)
                )
        self.theta = (
            post * self.pLL[None, :]
            + (1.0 - post) * self.pNL[None, :]
        )


class BayesFreshSky:
    """Bayesian-model-averaged belief with individual realized-energy queues."""

    name = "Bayes-FreshSky"

    def __init__(self, cfg):
        self.cfg = cfg
        self.filter = BayesianGEFilter(cfg.N)

    def __call__(self, A, oracle_belief, Q, w, pth, cfg):
        del oracle_belief
        score = cfg.V * w * A * self.filter.belief * B.PHI_LOS - Q * pth
        return B.topM(score, cfg)

    def update(self, energy):
        del energy

    def observe_feedback(self, attempted, ack):
        self.filter.step(attempted, ack)


class BayesFreshSkySafe(BayesFreshSky):
    """Bayes-FreshSky with a hard individual debt cap."""

    name = "Bayes-FreshSky-Safe"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.qcap = float(getattr(cfg, "qcap", 2.0))

    def __call__(self, A, oracle_belief, Q, w, pth, cfg):
        del oracle_belief
        score = cfg.V * w * A * self.filter.belief * B.PHI_LOS - Q * pth
        next_q = np.maximum(Q - cfg.pbar, 0.0) + pth
        score = np.where(next_q <= self.qcap + 1e-12, score, -1e18)
        return B.topM(score, cfg)


class BayesAggregateDPP(BayesFreshSky):
    """Same Bayesian filter but one aggregate budget queue."""

    name = "Bayes-Aggregate-DPP"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.Z = 0.0

    def __call__(self, A, oracle_belief, Q, w, pth, cfg):
        del oracle_belief, Q
        score = (
            cfg.V * w * A * self.filter.belief * B.PHI_LOS - self.Z * pth
        )
        return B.topM(score, cfg)

    def update(self, energy):
        self.Z = max(
            0.0,
            self.Z + float(np.sum(energy)) - self.cfg.N * self.cfg.pbar,
        )


def clone_cfg(**kwargs):
    cfg = B.clone(B.Cfg(), **kwargs)
    # AoI is unsaturated.  Amax only allocates finite-run histogram support.
    if "Amax" not in kwargs:
        cfg.Amax = int(cfg.T) + 1
    return cfg


def run_detailed(policy, cfg, seeds=(0, 1, 2, 3, 4)):
    """Matched hidden-channel runner with pathwise queue/violation metrics."""
    seed_rows = []
    for seed in seeds:
        rng = np.random.default_rng(7000 + seed)
        n_uav = cfg.N
        pol = policy(cfg) if isinstance(policy, type) else policy
        if hasattr(pol, "reset_seed"):
            pol.reset_seed(seed)
        pi_los, pth = B.geometry(cfg, rng)
        p_nl = 1.0 / cfg.mean_dwell
        p_ln = np.clip(p_nl * (1.0 - pi_los) / pi_los, 0.0, 1.0)
        B._ALPHA = 1.0 - p_ln
        B._BETA = 1.0 - p_nl
        los = rng.random(n_uav) < pi_los
        oracle_belief = pi_los.copy()
        age = np.ones(n_uav)
        queue = np.zeros(n_uav)
        event = rng.random(n_uav) < cfg.p_event
        weighted_age = 0.0
        count = 0
        energy_sum = np.zeros(n_uav)
        queue_max = np.zeros(n_uav)
        cumulative_surplus = np.zeros(n_uav)
        prefix_violation = np.zeros(n_uav)
        entropy_sum = 0.0
        entropy_count = 0
        for t in range(cfg.T):
            value = np.where(event, cfg.wev, 1.0)
            selected = pol(age, oracle_belief, queue, value, pth, cfg)
            energy = np.zeros(n_uav)
            ack = np.zeros(n_uav, dtype=bool)
            attempted = np.zeros(n_uav, dtype=bool)
            for n in selected:
                energy[n] = pth[n]
                attempted[n] = True
                success = B.PHI_LOS if los[n] else B.PHI_NLOS
                ack[n] = rng.random() < success
            if hasattr(pol, "update"):
                pol.update(energy)
            if hasattr(pol, "observe_feedback"):
                pol.observe_feedback(attempted, ack)
            age = np.where(ack, 1.0, age + 1.0)
            queue = np.maximum(queue - cfg.pbar, 0.0) + energy
            queue_max = np.maximum(queue_max, queue)
            cumulative_surplus += energy - cfg.pbar
            prefix_violation = np.maximum(prefix_violation, cumulative_surplus)
            # Oracle transition belief is supplied only to oracle baselines.
            post = oracle_belief.copy()
            for n in np.flatnonzero(attempted):
                p_ack = (
                    oracle_belief[n] * B.PHI_LOS
                    + (1.0 - oracle_belief[n]) * B.PHI_NLOS
                )
                if ack[n]:
                    post[n] = (
                        oracle_belief[n] * B.PHI_LOS / (p_ack + 1e-12)
                    )
                else:
                    post[n] = (
                        oracle_belief[n] * (1.0 - B.PHI_LOS)
                        / (1.0 - p_ack + 1e-12)
                    )
            oracle_belief = post * (1.0 - p_ln) + (1.0 - post) * p_nl
            u = rng.random(n_uav)
            los = np.where(los, u >= p_ln, u < p_nl)
            u = rng.random(n_uav)
            event = np.where(event, u >= cfg.ev_off, u < cfg.p_event)
            if t >= cfg.warmup:
                weighted_age += float(np.sum(value * age))
                energy_sum += energy
                count += 1
                if hasattr(pol, "filter"):
                    entropy_sum += float(np.mean(pol.filter.normalized_entropy))
                    entropy_count += 1
        seed_rows.append(
            dict(
                seed=seed,
                wAoI=weighted_age / count,
                mean_power=float(np.mean(energy_sum / count)),
                max_avg_power=float(np.max(energy_sum / count)),
                max_queue=float(np.max(queue_max)),
                max_prefix_violation=float(
                    np.max(np.maximum(prefix_violation, 0.0))
                ),
                mean_model_entropy=(
                    entropy_sum / entropy_count if entropy_count else np.nan
                ),
                feasible=bool(np.all(energy_sum / count <= cfg.pbar * 1.02)),
                certificate_ok=bool(
                    not hasattr(pol, "qcap")
                    or np.max(queue_max) <= pol.qcap + 1e-9
                ),
            )
        )
    frame = pd.DataFrame(seed_rows)
    summary = {
        "wAoI": frame.wAoI.mean(),
        "wAoI_std": frame.wAoI.std(ddof=1),
        "mean_power": frame.mean_power.mean(),
        "max_avg_power": frame.max_avg_power.mean(),
        "max_queue": frame.max_queue.max(),
        "max_prefix_violation": frame.max_prefix_violation.max(),
        "mean_model_entropy": frame.mean_model_entropy.mean(),
        "feasible_all_seeds": frame.feasible.all(),
        "certificate_all_seeds": frame.certificate_ok.all(),
    }
    return frame, summary


POLICIES = [
    ("Bayes-FreshSky", BayesFreshSky),
    ("Bayes-Aggregate-DPP", BayesAggregateDPP),
    ("TS-DPP adapted", B.TSDPPAdapted26),
    ("FreshSky-B oracle", B.belief_whittle),
    ("Wang'26 oracle", B.wang26),
    ("Memoryless", B.memoryless),
]


def main():
    os.makedirs(B.RES, exist_ok=True)
    summaries = []
    seed_frames = []
    for qcap in [0.5, 1.0, 2.0, 4.0]:
        cfg = clone_cfg(pbar=0.4, qcap=qcap)
        frame, summary = run_detailed(BayesFreshSkySafe, cfg)
        summary.update(policy="Bayes-FreshSky-Safe", qcap=qcap)
        summaries.append(summary)
        seed_frames.append(frame.assign(policy="Bayes-FreshSky-Safe", qcap=qcap))
        print(
            f"Safe qcap={qcap:3.1f}: W-AoI={summary['wAoI']:7.1f}"
            f"+/-{summary['wAoI_std']:5.1f} maxP={summary['max_avg_power']:.3f} "
            f"maxQ={summary['max_queue']:.3f} cert={summary['certificate_all_seeds']}"
        )
    cfg = clone_cfg(pbar=0.4)
    for name, policy in POLICIES:
        frame, summary = run_detailed(policy, cfg)
        summary.update(policy=name, qcap=np.nan)
        summaries.append(summary)
        seed_frames.append(frame.assign(policy=name, qcap=np.nan))
        print(
            f"{name:22s}: W-AoI={summary['wAoI']:7.1f}"
            f"+/-{summary['wAoI_std']:5.1f} maxP={summary['max_avg_power']:.3f} "
            f"maxQ={summary['max_queue']:.3f} feasible={summary['feasible_all_seeds']}"
        )
    pd.DataFrame(summaries).to_csv(
        os.path.join(B.RES, "bayes_hmm_summary.csv"), index=False
    )
    pd.concat(seed_frames, ignore_index=True).to_csv(
        os.path.join(B.RES, "bayes_hmm_seeds.csv"), index=False
    )
    print("saved bayes_hmm_summary.csv and bayes_hmm_seeds.csv")


if __name__ == "__main__":
    main()
