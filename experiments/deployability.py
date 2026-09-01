# -*- coding: utf-8 -*-
"""THE DEPLOYABILITY PLANE.

Every scheduler is scored on the two axes an operator actually cares about:
  x  steady-state weighted AoI          -- freshness  (what every paper optimizes)
  y  peak cumulative budget violation
     under a mid-mission distribution
     shift, in mW*slots                 -- safety     (what almost no paper measures)

Both axes are measured for every known-channel scheduler under the identical shift and
the identical 5 seeds already used for Table III, so the plane is apples-to-apples.

Nothing here is tuned to flatter FreshSky: the fixed price is granted the most
aggressive price that is still feasible BEFORE the shift (its best honest
setting), and the Deep Index Policy is fully trained (cached to drl_model.pt).

Emits results/deployability.csv and figs/fig_deploy.png.
"""
import os, numpy as np, pandas as pd, torch
import freshsky_sim as F
import robustness_shift as RS

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, 'results'), os.path.join(HERE, 'figs')
CKPT = os.path.join(HERE, 'drl_model.pt')

T, TS = 3000, 1500
SEEDS = (0, 1, 2, 3, 4)
base = F.Cfg().derived()
BUD = base.pbar
SHIFT = dict(p_ihigh=0.6, I_high=4e-8, p_event=0.5, mean_sojourn_ev=50.0)

# what each scheduler needs before it can be deployed (hard, checkable facts)
NEEDS = {   # policy-name prefix -> (needs offline tuning/training?, value-aware?)
    'Round-Robin':        (False, False), 'Max-Age-First':    (False, False),
    'Max-Weight':         (False, True),  'Whittle-AoI':      (False, False),
    'Opportunistic':      (False, False), 'VoI-Greedy':       (False, True),
    'AoII-Whittle':       (False, True),  'CW-Whittle':       (False, True),
    'Lyap-DPP':           (False, False), 'Cost-AoI-Whittle': (False, False),
    'Fixed-Price':        (True,  True),  'AoI-Energy Online':(True,  True),
    'Tang-CMDP':          (True,  False),
    'Deep Index':         (True,  True),  'FreshSky':         (False, True),
}
def needs(p):
    for k, v in NEEDS.items():
        if p.startswith(k): return v
    return (False, False)


def peak_cum_violation(pp):
    """Prop-3 quantity: peak cumulative over-budget energy from the shift onward."""
    return max(0.0, float(np.cumsum(pp[TS:] - BUD).max()))


def measure(cls, V=200.0):
    pcv, post = [], []
    for s in SEEDS:
        _, pp = RS.run_shift(cls, F.clone(base, V=V, seed=s), TS, T, SHIFT)
        pcv.append(peak_cum_violation(pp)); post.append(pp[TS + 500:].mean())
    return float(np.mean(pcv)), float(np.std(pcv)), float(np.mean(post))


# ---- Deep Index Policy: train once, cache (so this figure is cheap to rebuild) ----
if os.path.exists(CKPT):
    net = F.DRLNet(); net.load_state_dict(torch.load(CKPT)); F.DRL_MODEL = net
    print(f"loaded cached Deep Index Policy <- {CKPT}")
else:
    print("training Deep Index Policy (~20 min, once; then cached)...")
    F.DRL_MODEL = F.train_drl(base)
    torch.save(F.DRL_MODEL.state_dict(), CKPT)
    print(f"cached -> {CKPT}")

# ---- fixed price: offline-tuned on the PRE-shift regime (its best honest setting) ----
for lam in [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    F.FixedPriceThr.LAM = lam
    _, pa = RS.run_shift(F.FixedPriceThr, F.clone(base, V=200.0), T + 10, TS, {})
    if pa.mean() <= BUD:
        break
print(f"offline-tuned fixed price lambda* = {lam:.2f}\n")

su = pd.read_csv(os.path.join(RES, 'summary_default.csv'))
POLS = F.ANALYTIC + [F.DRLPolicy]
rows = []
for P in POLS:
    if P is F.FreshSkyWhittle:
        continue                                   # identical to FreshSkyDPP (shares its row)
    v, sd, post = measure(P)
    m = su[su['policy'] == P.name]
    if not len(m):
        print(f"  !! {P.name} missing from summary_default.csv -- skipped"); continue
    m = m.iloc[0]
    tune, val = needs(P.name)
    rows.append(dict(policy=P.name, wAoI=m['wAoI'], wAoI_std=m['wAoI_std'],
                     power=m['max_avg_power'], steady_feasible=bool(m['energy_ok']),
                     peak_cum_viol=v, pcv_std=sd, post_shift_power=post,
                     recovers=post <= BUD * 1.02, needs_tuning=tune, value_aware=val))
    print(f"  {P.name[:40]:40s} AoI={m['wAoI']:6.1f}  viol={v:8.0f}  "
          f"recovers={rows[-1]['recovers']}  tuning={tune}")

df = pd.DataFrame(rows).sort_values('peak_cum_viol')
df.to_csv(os.path.join(RES, 'deployability.csv'), index=False)
print(f"\nwrote {os.path.join(RES,'deployability.csv')}")

fs = df[df['policy'].str.startswith('FreshSky')].iloc[0]
print(f"\nFreshSky: AoI {fs['wAoI']:.1f}, shift violation {fs['peak_cum_viol']:.0f}")
for _, r in df.iterrows():
    if r['policy'].startswith('FreshSky'): continue
    fresher = r['wAoI'] < fs['wAoI']
    safer = r['peak_cum_viol'] < fs['peak_cum_viol']
    tag = 'DOMINATES US' if (fresher and safer) else \
          ('fresher but less safe' if fresher else ('safer but staler' if safer else 'DOMINATED by us'))
    print(f"  {r['policy'][:38]:38s} {tag}")
