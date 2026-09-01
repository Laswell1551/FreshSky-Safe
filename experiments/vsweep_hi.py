# -*- coding: utf-8 -*-
"""Extend the V-sweep into the high-V (budget-approaching) regime, using the SAME
5 seeds as the main table (summary_default.csv) so FreshSky's curve and its table
row are the same measurement. Answers: what is the best ENERGY-FEASIBLE operating
point of FreshSky, and does it beat the strongest feasible baseline?"""
import os, pandas as pd
import freshsky_sim as F

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
SEEDS = (0, 1, 2, 3, 4)          # identical to run_multi's default / the main table

base = F.Cfg().derived()
print(f"budget pbar = {base.pbar:.2f} mW   (feasible iff max per-UAV avg power <= pbar*1.02)")
print(f"seeds = {SEEDS}\n")

rows = []
for V in [50, 100, 200, 500, 1000, 2000, 5000]:
    r = F.run_multi(F.FreshSkyDPP, F.clone(base, V=float(V)), seeds=SEEDS)
    rows.append(dict(V=V, wAoI=r['wAoI'], wAoI_std=r['wAoI_std'],
                     max_avg_power=r['max_avg_power'], Qbacklog=r['Qbacklog'],
                     det_latency=r['det_latency'], peakAoI=r['peakAoI'],
                     energy_ok=r['energy_ok']))
    print(f"  V={V:5d}  wAoI={r['wAoI']:6.2f} +/-{r['wAoI_std']:4.2f}   "
          f"pow={r['max_avg_power']:.3f} mW   Q={r['Qbacklog']:7.0f}   "
          f"feasible={r['energy_ok']}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, 'vsweep_hi.csv'), index=False)

# --- the decisive comparison: strongest ONLINE feasible baseline ---
s = pd.read_csv(os.path.join(RES, 'summary_default.csv'))
feas = s[s['energy_ok']].sort_values('wAoI')
print("\n--- energy-feasible baselines (5 seeds, from summary_default.csv) ---")
for _, r in feas.iterrows():
    print(f"  {r['policy'][:42]:42s} wAoI={r['wAoI']:6.2f} +/-{r['wAoI_std']:4.2f}  pow={r['max_avg_power']:.3f} mW")

best_feas = df[df['energy_ok']].sort_values('wAoI').iloc[0]
print(f"\n>>> FreshSky best FEASIBLE point: V={int(best_feas['V'])}  "
      f"wAoI={best_feas['wAoI']:.2f}  pow={best_feas['max_avg_power']:.3f} mW")
rivals = feas[~feas['policy'].str.contains('ours')]
if len(rivals):
    b = rivals.iloc[0]
    print(f">>> best feasible BASELINE:      {b['policy']}  wAoI={b['wAoI']:.2f}  pow={b['max_avg_power']:.3f} mW")
    d = 100 * (b['wAoI'] - best_feas['wAoI']) / b['wAoI']
    verdict = "FreshSky WINS" if d > 0 else "FreshSky LOSES"
    print(f">>> {verdict} by {abs(d):.1f}% on weighted AoI, at "
          f"{best_feas['max_avg_power']:.2f} vs {b['max_avg_power']:.2f} mW "
          f"(budget {base.pbar:.1f})")
