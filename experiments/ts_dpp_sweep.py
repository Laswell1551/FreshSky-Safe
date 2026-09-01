# -*- coding: utf-8 -*-
"""Correlation sweep for the learned-transition TS-DPP adaptation.

This script keeps the hidden-channel harness, seeds, geometry distribution,
feedback, and budget identical across policies.  It writes seed-aggregated
real results only; no paper number is hard-coded.
"""
import os
import pandas as pd
import belief_whittle as B

DWELL = [1.5, 2, 3, 5, 8, 14, 20]
POLICIES = [
    ("FreshSky-B", B.belief_whittle),
    ("Belief-DPP oracle", B.BeliefDPP26),
    ("TS-DPP adapted", B.TSDPPAdapted26),
    ("Memoryless", B.memoryless),
]


def main():
    rows = []
    for dwell in DWELL:
        cfg = B.clone(B.Cfg(), mean_dwell=dwell)
        for name, policy in POLICIES:
            result = B.run(policy, cfg, details=True)
            row = {"mean_nlos_dwell": dwell, "policy": name, **result}
            rows.append(row)
            print(
                f"dwell={dwell:4.1f} {name:18s} "
                f"W-AoI={result['wAoI']:7.1f}+/-{result['wAoI_std']:5.1f} "
                f"maxP={result['max_power']:.3f} feasible={result['feasible']}"
            )
    os.makedirs(B.RES, exist_ok=True)
    path = os.path.join(B.RES, "ts_dpp_sweep.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"saved {path}")


if __name__ == "__main__":
    main()

