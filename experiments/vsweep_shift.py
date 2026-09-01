# -*- coding: utf-8 -*-
"""Does FreshSky's shift-robustness SURVIVE at the high-V operating point?

Thm 2 says the backlog is O(V) and Prop 3's finite-time violation bound is
Q^max ~ V, so pushing V up for steady-state freshness must cost transient safety.
This quantifies that tradeoff on the SAME mid-mission shift used in the paper,
and checks FreshSky at each V against the two energy-aware baselines.

Reported per policy:
  post-shift steady power  -- does it RECOVER feasibility? (<= budget)
  peak cumulative violation -- the Prop-3 quantity: max_t sum_{tau=t_shift}^{t}
                               (p(tau) - pbar), floored at 0 (mW-slots)
"""
import os, numpy as np, pandas as pd
import freshsky_sim as F
import robustness_shift as RS

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
T, TS = 3000, 1500
base = F.Cfg().derived()
bud = base.pbar
shift = dict(p_ihigh=0.6, I_high=4e-8, p_event=0.5, mean_sojourn_ev=50.0)
SEEDS = (0, 1, 2, 3, 4)


def peak_cum_violation(pp):
    """Prop-3 quantity: peak cumulative over-budget energy after the shift."""
    d = np.cumsum(pp[TS:] - bud)
    return max(0.0, float(d.max()))


def measure(cls, V):
    post, pcv = [], []
    for s in SEEDS:
        cfg = F.clone(base, V=float(V), seed=s)
        _, pp = RS.run_shift(cls, cfg, TS, T, shift)
        post.append(pp[TS + 500:].mean())
        pcv.append(peak_cum_violation(pp))
    return float(np.mean(post)), float(np.mean(pcv)), float(np.std(pcv))


# offline-tune the fixed price on the PRE-shift regime (unchanged protocol)
for lam in [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    F.FixedPriceThr.LAM = lam
    _, pa = RS.run_shift(F.FixedPriceThr, F.clone(base, V=200.0), T + 10, TS, {})
    if pa.mean() <= bud:
        break
print(f"offline-tuned fixed price lambda* = {lam:.2f}")
print(f"budget = {bud:.2f} mW,  shift at t={TS}, {len(SEEDS)} seeds\n")

rows = []
p, v, sd = measure(F.FixedPriceThr, 200.0)
rows.append(dict(policy='Fixed-Price (offline-tuned)', V=np.nan, post_power=p,
                 peak_cum_viol=v, pcv_std=sd, recovers=p <= bud * 1.02))
p, v, sd = measure(F.AoIEnergyOnline, 200.0)
rows.append(dict(policy="Abd-Elmagid'25 (online)", V=np.nan, post_power=p,
                 peak_cum_viol=v, pcv_std=sd, recovers=p <= bud * 1.02))
for V in [200, 500, 1000, 2000]:
    p, v, sd = measure(F.FreshSkyDPP, V)
    rows.append(dict(policy=f'FreshSky (V={V})', V=V, post_power=p,
                     peak_cum_viol=v, pcv_std=sd, recovers=p <= bud * 1.02))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, 'vsweep_shift.csv'), index=False)
print(f"{'policy':30s} {'post-shift pow':>15s} {'recovers?':>10s} {'peak cum. viol.':>17s}")
print('-' * 76)
for _, r in df.iterrows():
    print(f"{r['policy']:30s} {r['post_power']:12.2f} mW {str(r['recovers']):>10s} "
          f"{r['peak_cum_viol']:12.0f} +/-{r['pcv_std']:.0f}")
