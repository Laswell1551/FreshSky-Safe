# -*- coding: utf-8 -*-
"""FreshSky-L: learned hidden-channel transitions plus individual energy debt.

The transition learner and ACK/NACK feedback are intentionally identical to
TSDPPAdapted26.  The controlled difference is the constraint state:

* TS-DPP adapted: one aggregate budget queue Z.
* FreshSky-L: one realized energy-debt queue Q_n per UAV.

This module isolates the effect of replacing aggregate debt with individual
realized energy-debt queues.
"""
import os
import pandas as pd
import belief_whittle as B


class FreshSkyLearned(B.TSDPPAdapted26):
    """Learned-transition FreshSky with individual per-UAV energy queues."""

    def __call__(self, A, bel, Q, w, pth, cfg):
        del bel  # never consume the harness's true-transition belief
        score = cfg.V * w * A * self.th * B.PHI_LOS - Q * pth
        return B.topM(score, cfg)

    def update(self, e):
        # The harness updates the individual Q_n queues after every slot.
        # There is deliberately no aggregate queue in this policy.
        del e


POLICIES = [
    ("FreshSky-L learned", FreshSkyLearned),
    ("FreshSky-B oracle-transition", B.belief_whittle),
    ("Belief-DPP oracle-transition", B.BeliefDPP26),
    ("TS-DPP adapted learned", B.TSDPPAdapted26),
    ("Wang'26 oracle-transition", B.wang26),
    ("Memoryless", B.memoryless),
]


def run_budget(name, pbar):
    cfg = B.clone(B.Cfg(), pbar=pbar)
    rows = []
    print(f"== {name}: pbar={pbar} mW per UAV ==")
    for policy_name, policy in POLICIES:
        result = B.run(policy, cfg, details=True)
        rows.append({"setting": name, "pbar": pbar,
                     "policy": policy_name, **result})
        print(
            f"{policy_name:30s} W-AoI={result['wAoI']:7.1f}"
            f"+/-{result['wAoI_std']:5.1f} maxP={result['max_power']:.3f} "
            f"feasible={result['feasible']}"
        )
    return rows


def run_correlation_sweep():
    rows = []
    for dwell in [1.5, 2, 3, 5, 8, 14, 20]:
        cfg = B.clone(B.Cfg(), mean_dwell=dwell, pbar=0.4)
        for policy_name, policy in POLICIES:
            result = B.run(policy, cfg, details=True)
            rows.append({"mean_nlos_dwell": dwell, "pbar": cfg.pbar,
                         "policy": policy_name, **result})
            print(
                f"dwell={dwell:4.1f} {policy_name:30s} "
                f"W-AoI={result['wAoI']:7.1f} maxP={result['max_power']:.3f} "
                f"feasible={result['feasible']}"
            )
    return rows


def main():
    os.makedirs(B.RES, exist_ok=True)
    budget_rows = run_budget("default", 1.6)
    budget_rows += run_budget("binding-individual", 0.4)
    pd.DataFrame(budget_rows).to_csv(
        os.path.join(B.RES, "learned_belief_budget.csv"), index=False)
    sweep_rows = run_correlation_sweep()
    pd.DataFrame(sweep_rows).to_csv(
        os.path.join(B.RES, "learned_belief_correlation.csv"), index=False)
    print("saved learned_belief_budget.csv and learned_belief_correlation.csv")


if __name__ == "__main__":
    main()
