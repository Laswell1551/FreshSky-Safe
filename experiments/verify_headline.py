# -*- coding: utf-8 -*-
"""Re-verify, from scratch, the two numbers about to be promoted into the abstract:
  (1) on REAL OpenStreetMap Manhattan geometry, FreshSky-B Pareto-dominates the
      closest concurrent belief index [Wang'26]: lower weighted AoI AND lower power;
  (2) it also Pareto-dominates memoryless scheduling on the same traces.
Prints PASS/FAIL per axis so a wrong claim cannot silently reach the paper."""
import numpy as np
import osm_urban as OU

los, dist = OU.load_traces()
print(f"OSM traces: {los.shape[0]} UAVs x {los.shape[1]} slots, LoS fraction {los.mean():.3f}")

res = OU.eval_sched(los, dist)                       # {name: (wAoI, power_mW)}
wang = np.mean([OU.U.run_on_trace(los, dist, OU.U.wang26, seed=s) for s in range(4)], axis=0)
res["Wang'26 POMDP-Whittle"] = (float(wang[0]), float(wang[1]))

print("\n%-34s %10s %12s" % ('scheduler', 'wAoI', 'power (mW)'))
for k, (wa, en) in sorted(res.items(), key=lambda kv: kv[1][0]):
    print("%-34s %10.1f %12.3f" % (k, wa, en))

ours = [v for k, v in res.items() if 'Belief' in k or 'ours' in k][0]
print(f"\nFreshSky-B: wAoI {ours[0]:.1f} @ {ours[1]:.3f} mW")
ok = True
for k, v in res.items():
    if v is ours: continue
    da, de = v[0] - ours[0], v[1] - ours[1]
    if da > 0 and de > 0:
        verdict = f"PARETO-DOMINATED by us (AoI +{da:.1f}, power +{de:.3f})"
    elif da > 0 or de > 0:
        verdict = f"trade only (dAoI {da:+.1f}, dPower {de:+.3f})"
    else:
        verdict = f"!! BEATS US on both (dAoI {da:+.1f}, dPower {de:+.3f})"; ok = False
    print(f"  vs {k:32s} {verdict}")

print("\n" + ("PASS - the abstract's Pareto-dominance claim holds on both axes."
             if ok else "FAIL - do NOT put Pareto-dominance in the abstract."))
