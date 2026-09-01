# -*- coding: utf-8 -*-
"""SECOND real-city zero-shot anchor (generalization beyond Manhattan).

Runs the EXACT same pipeline as osm_urban.py -- real OpenStreetMap building
footprints -> 3D height grid -> geometric UAV->GS LoS ray-march -> the UNCHANGED
paper belief-Whittle / memoryless / Wang'26 schedulers -- on a SECOND, structurally
different real city: the City of London financial core (a tall cluster: 22
Bishopsgate, the Gherkin, Cheesegrater, Heron Tower) set in an IRREGULAR, non-grid
medieval street layout -- the opposite of Manhattan's regular supertall grid.

We reuse osm_urban's functions verbatim by overriding only its city bbox/slug/cache
globals, so the processing code is provably identical; only the geography changes.
This is the zero-shot test on a different urban morphology and a different
continent. Result reported honestly (dominates vs trades), like Manhattan.

Building data (C) OpenStreetMap contributors, ODbL. Raw response cached to
sim/data/osm_city_of_london.json for offline reproduction.
"""
import os, numpy as np
import osm_urban as O          # reuse fetch/parse/grid/place_gs/gen_traces/eval UNCHANGED
import urban_blockage as U

# ---- override ONLY the geography (a ~1.1 km City-of-London core) -------------
O.CITY = "City of London financial core (Bishopsgate cluster), London, UK"
O.CITY_SHORT = "City of London"
O.SLUG = "city_of_london"
O.MINLAT, O.MINLON, O.MAXLAT, O.MAXLON = 51.5100, -0.0900, 51.5200, -0.0740
O.CACHE = os.path.join(O.DATA, "osm_city_of_london.json")
TRACE = os.path.join(O.DATA, "osm_traces_city_of_london.npz")

print("=" * 74)
print(f"SECOND real-city anchor (OSM footprints):  {O.CITY}")
print(f"bbox (S,W,N,E) = ({O.MINLAT}, {O.MINLON}, {O.MAXLAT}, {O.MAXLON})")
print("Building data (C) OpenStreetMap contributors, ODbL")
print("=" * 74)

data, origin = O.fetch_or_load()
polys, heights, src = O.parse_buildings(data)
nb = len(polys)
if nb == 0:
    raise SystemExit("[FATAL] no buildings parsed; check bbox / cache.")
ntag = src['tagged']
print(f"[buildings] {nb} footprints  (source: {origin})")
print(f"[heights]  tagged={ntag} ({100*ntag/nb:.1f}%) | "
      f"levelsx3={src['levels']} | default={src['default']} | "
      f"median={np.median(heights):.1f} m  p90={np.percentile(heights,90):.1f} m  max={heights.max():.1f} m")

H, cell, sx, sy, _ = O.build_grid(polys, heights)
gs, (gi, gj, gh) = O.place_gs(H, cell)
print(f"[grid]     {H.shape[0]}x{H.shape[1]} @ {cell:.0f} m | span {sx:.0f}x{sy:.0f} m | "
      f"built-up {(H>0).mean()*100:.1f}% | GS roof {gh:.0f} m, z={gs[2]:.1f} m")

if os.path.exists(TRACE):
    z = np.load(TRACE); los, dist = z['los'], z['dist']
    print(f"[traces]   loaded cached {os.path.relpath(TRACE, O.HERE)} -> offline")
else:
    print(f"[traces]   ray-marching UAV->GS LoS over REAL height grid (N={O.N_UAV}, T={O.T_SLOTS})...")
    los, dist = O.gen_traces_osm(H, cell, sx, sy, gs)
    np.savez_compressed(TRACE, los=los, dist=dist)
lf = float(los.mean()); ac = O.autocorr_avg(los)
print(f"[LoS]      LoS fraction = {lf:.3f}  (NLoS {1-lf:.3f})")
print(f"[autocorr] lag1={ac[0]:.3f}  lag5={ac[4]:.3f}  lag10={ac[9]:.3f}  (correlated -> non-Markov)")

# ---- scheduler comparison: SAME paper code, incl. Wang'26 -------------------
res = O.eval_sched(los, dist)
waw, enw = np.mean([U.run_on_trace(los, dist, U.wang26, seed=s) for s in O.RUN_SEEDS], axis=0)
res["Wang'26 POMDP"] = (float(waw), float(enw))
for name in ("Belief-Whittle (ours)", "Memoryless", "Wang'26 POMDP"):
    wa, en = res[name]
    print(f"[sched]    {name:22s} wAoI={wa:8.1f}   power/UAV={en:.3f} mW")
bw, ml, dAoI, dPow, dom = O.dominance(res)
print(f"[RESULT]   vs Memoryless: AoI {dAoI:+.1f}%, power {dPow:+.1f}%  -> "
      f"{'Pareto-DOMINATES' if dom else 'Pareto-TRADES'}")
# vs Wang'26 (the closest concurrent work)
dAw = 100*(waw-bw[0])/waw; dPw = 100*(enw-bw[1])/enw
domW = (bw[0] <= waw and bw[1] <= enw) and (bw[0] < waw or bw[1] < enw)
print(f"[RESULT]   vs Wang'26:    AoI {dAw:+.1f}%, power {dPw:+.1f}%  -> "
      f"{'Pareto-DOMINATES' if domW else 'Pareto-TRADES'}")

print("[robust]   across independent trajectory (geometry) seeds:")
for gs_seed in (0, 1, 2):
    if gs_seed == O.GEO_SEED and os.path.exists(TRACE):
        lo, di = los, dist
    else:
        lo, di = O.gen_traces_osm(H, cell, sx, sy, gs, seed=gs_seed)
    _, _, dA, dP, dm = O.dominance(O.eval_sched(lo, di))
    print(f"             geo seed {gs_seed}: LoS={float(lo.mean()):.3f}  "
          f"dAoI={dA:+5.1f}%  dPow={dP:+5.1f}%  dominates={dm}")
print("=" * 74)
print("Building data (C) OpenStreetMap contributors, ODbL. Cached for offline reproduction.")
