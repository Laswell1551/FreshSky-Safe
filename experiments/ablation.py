# -*- coding: utf-8 -*-
"""Component ablation of FreshSky-B (belief-Whittle), reusing the SAME
air-ground partial-observability harness as belief_whittle.py. Under a tight
energy budget we remove one ingredient at a time from the index
B_n = V w_n A_n theta_n phi_L - Q_n p_th:
  Full        : value w, belief theta, energy queue Q  (all on)
  - value     : w -> 1            (freshness no longer value-weighted)
  - belief    : theta -> 1        (no LoS belief; assume always deliverable)
  - energy Q  : drop Q p_th term  (no energy awareness)
Emits a CSV; the paper's ablation table is generated from these numbers."""
import os, numpy as np, csv
import belief_whittle as B
PHI=B.PHI_LOS
RES=os.path.join(os.path.dirname(__file__),"results")

def full(A,bel,Q,w,pth,cfg):      return B.topM(cfg.V*w*A*bel*PHI - Q*pth, cfg)
def no_value(A,bel,Q,w,pth,cfg):  return B.topM(cfg.V*1.0*A*bel*PHI - Q*pth, cfg)
def no_belief(A,bel,Q,w,pth,cfg): return B.topM(cfg.V*w*A*1.0*PHI - Q*pth, cfg)
def no_queue(A,bel,Q,w,pth,cfg):  return B.topM(cfg.V*w*A*bel*PHI, cfg)

if __name__=="__main__":
    cfg=B.clone(B.Cfg(), V=30.0, pbar=0.22)     # tight budget so each component binds
    bud=cfg.pbar
    print(f"== FreshSky-B component ablation (tight budget {bud:.2f} mW/UAV, V={cfg.V:.0f}) ==")
    rows=[]
    for name,pol in [("Full FreshSky-B",full),("$-$ value weight",no_value),
                     ("$-$ belief",no_belief),("$-$ energy queue",no_queue)]:
        wa,en=B.run(pol,cfg)
        feas=en<=bud*1.02
        rows.append((name,wa,en,feas))
        print(f"  {name:18s} wAoI={wa:6.1f}  power/UAV={en:.3f} mW  feasible={feas}")
    with open(os.path.join(RES,"ablation.csv"),"w",newline="") as f:
        wr=csv.writer(f); wr.writerow(["variant","wAoI","power_per_uav_mW","feasible"])
        for name,wa,en,feas in rows: wr.writerow([name,f"{wa:.1f}",f"{en:.3f}",feas])
    print("saved results/ablation.csv")
    # Wang'26 baseline at the SAME tight budget (value-/energy-agnostic -> violates): scripts the
    # binding-budget number quoted in the belief-eval text; kept out of ablation.csv (it is a baseline).
    wa,en=B.run(B.wang26,cfg); feas=en<=bud*1.02
    print(f"  {'Wang26 [baseline]':18s} wAoI={wa:6.1f}  power/UAV={en:.3f} mW  feasible={feas}  (cap {bud} mW)")
    with open(os.path.join(RES,"wang26_budget.csv"),"w",newline="") as f:
        csv.writer(f).writerows([["method","wAoI","power_per_uav_mW","cap_mW","feasible"],
                                 ["Wang26",f"{wa:.1f}",f"{en:.3f}",f"{bud}",feas]])
    print("saved results/wang26_budget.csv")
