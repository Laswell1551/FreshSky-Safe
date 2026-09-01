# -*- coding: utf-8 -*-
# Academic Figure Skill Asset Confirmation: LineTrend plus compact scatter ->
# PARAM-INHERIT. Only typography, palette, open axes, and uncertainty grammar
# are inherited; map geometry and scheduler values come from cached real data.
"""
REAL-DATA urban generalization anchor for FreshSky-B.

Instead of a procedurally generated city (sim/urban_blockage.py), this script
downloads REAL OpenStreetMap building footprints for a NAMED, dense city
district, rasterizes them into a 3D height grid, and generates UAV->ground
line-of-sight (LoS)/NLoS blockage traces by exact geometric ray-marching along
straight patrol trajectories. It then runs the SAME belief-Whittle scheduler
and air-ground channel model as the paper (imported UNCHANGED from
urban_blockage.py) on these real-geometry traces. This is a zero-shot /
no-retraining transfer test: FreshSky-B is DESIGNED assuming a two-state Markov
blockage channel, yet is EVALUATED here on real, strongly-correlated non-Markov
blockage produced by real geographic building data.

The district is a ~1.2 km Midtown Manhattan CORE (Rockefeller Center / Bryant
Park / Times Square), sized so a single ground station's service area matches
the paper's operational scale (its procedural city is 800 m; its link geometry
uses UAV-GS ranges <=380 m). At this operational scale FreshSky-B Pareto-
dominates the memoryless scheduler on the REAL geometry. As an HONEST boundary,
that dominance narrows monotonically with the LoS fraction and finally turns
into a Pareto TRADE (slightly lower power, slightly higher AoI) only in the
low-LoS limit reached by an atypically large ~1.9 km single-GS span, where the
UAV-GS links cross a canyon of supertalls and are blocked >85% of the time.
This boundary is disclosed, not hidden (see the final report).

Building data (C) OpenStreetMap contributors, ODbL (https://www.openstreetmap.org/copyright).
Downloaded via the Overpass API; the raw response is cached to
sim/data/osm_<city>.json so this script REPRODUCES OFFLINE from cache.

Run:  /e/Software/Anaconda/python osm_urban.py
Deps: numpy (1.26.4), matplotlib, requests. No geopandas/shapely/osmnx.
"""
import os, re, json, time, math
import numpy as np
import requests
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.path import Path

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'],
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.bbox': 'tight', 'savefig.dpi': 300,
})
BLUE, RED, GREEN, ORANGE, PURPLE, GREY, BLACK = (
    '#2166AC', '#B2182B', '#1B7837', '#F1A340', '#762A83', '#999999', '#222222')

import urban_blockage as U      # REUSE channel constants + run_on_trace + policies UNCHANGED

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); FIG = os.path.join(HERE, "figs")
os.makedirs(DATA, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# ------------------------- named district + real bbox -------------------------
CITY = "Midtown Manhattan (Rockefeller Center core), New York City"
CITY_SHORT = "Midtown Manhattan"
SLUG = "midtown_manhattan"
# Overpass / standard bbox order = (south, west, north, east) = (minlat,minlon,maxlat,maxlon)
# ~1.2 km dense Midtown core (approx. W44th-W54th St, 5th-7th Ave), centered near
# Rockefeller Center (40.7539 N, 73.9840 W); matches the paper's operational scale.
MINLAT, MINLON, MAXLAT, MAXLON = 40.7485, -73.9911, 40.7593, -73.9769
CACHE = os.path.join(DATA, f"osm_{SLUG}.json")

CELL = 6.0        # height-grid cell size (m); task target 5-8 m
N_UAV = 12; T_SLOTS = 2000; GEO_SEED = 1; RUN_SEEDS = range(4)   # match e12_urban.py

# ------------------------- Overpass fetch (cache + mirrors + retry) -----------
def fetch_or_load():
    """Load cached raw JSON if present (offline-reproducible); else query
    Overpass (two mirrors, 3 attempts, 20 s backoff) and cache the raw JSON."""
    if os.path.exists(CACHE):
        with open(CACHE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[cache]    loaded {os.path.relpath(CACHE, HERE)} "
              f"({os.path.getsize(CACHE)//1024} KiB) -> OFFLINE reproduction")
        return data, 'cache'
    q = (f'[out:json][timeout:60];'
         f'(way["building"]({MINLAT},{MINLON},{MAXLAT},{MAXLON}););out body geom;')
    mirrors = ['https://overpass-api.de/api/interpreter',
               'https://overpass.kumi.systems/api/interpreter']
    last = None
    for attempt in range(3):
        for url in mirrors:
            try:
                print(f"[overpass] POST {url} (attempt {attempt+1}/3)")
                r = requests.post(url, data={'data': q}, timeout=120,
                                  headers={'User-Agent':
                                           'freshsky-research/1.0 (OSM building footprints; ODbL)'})
                if r.status_code == 200:
                    data = r.json()
                    with open(CACHE, 'w', encoding='utf-8') as f:
                        json.dump(data, f)
                    print(f"[overpass] OK -> cached {os.path.relpath(CACHE, HERE)} "
                          f"({len(r.content)//1024} KiB)")
                    return data, 'overpass'
                print(f"[overpass] HTTP {r.status_code} from {url}")
                last = f"HTTP {r.status_code}"
            except Exception as e:
                print(f"[overpass] ERROR {url}: {e!r}")
                last = repr(e)
        if attempt < 2:
            print("[overpass] all mirrors failed; waiting 20 s then retrying...")
            time.sleep(20)
    raise SystemExit(
        f"[FATAL] Overpass unreachable ({last}) and no local cache present. "
        f"Re-run when online; the raw data will then be cached at {CACHE} for "
        f"offline reuse. (No data fabricated.)")

# ------------------------- building parsing -----------------------------------
_NUM = re.compile(r'-?\d+(?:\.\d+)?')
def _first_float(s):
    m = _NUM.search(str(s)); return float(m.group()) if m else None

def parse_height(tags):
    """Return (height_m, source) with source in {'tagged','levels','default'}.
    Priority: explicit 'height' tag (meters; feet converted) > building:levels*3 m
    > 10 m default. Values clipped to a sane [2,500] m."""
    if 'height' in tags:
        raw = str(tags['height']).strip().lower().replace(',', '.')
        v = _first_float(raw)
        if v is not None:
            if ('ft' in raw) or ("'" in raw) or ('feet' in raw):
                v *= 0.3048
            return float(np.clip(v, 2.0, 500.0)), 'tagged'
    if 'building:levels' in tags:
        v = _first_float(tags['building:levels'])
        if v is not None and v > 0:
            return float(np.clip(v * 3.0, 2.0, 500.0)), 'levels'
    return 10.0, 'default'

def parse_buildings(data):
    ways = [e for e in data.get('elements', [])
            if e.get('type') == 'way' and e.get('geometry')]
    polys, heights, src = [], [], {'tagged': 0, 'levels': 0, 'default': 0}
    for e in ways:
        g = e['geometry']
        if len(g) < 3:
            continue
        lat = np.array([p['lat'] for p in g], float)
        lon = np.array([p['lon'] for p in g], float)
        h, s = parse_height(e.get('tags', {}))
        polys.append((lat, lon)); heights.append(h); src[s] += 1
    return polys, np.array(heights, float), src

# ------------------------- projection + rasterization -------------------------
def build_grid(polys, heights):
    """Equirectangular projection about the bbox-center latitude (origin at the
    SW corner so all coords >=0), then rasterize building footprints to a height
    grid H[ix,iy] = max building height over cell centers inside the footprint.
    Indexing matches urban_blockage.los_blocked (i=x//cell, j=y//cell)."""
    lat_c = 0.5 * (MINLAT + MAXLAT)
    kx = 111320.0 * math.cos(math.radians(lat_c)); ky = 110540.0
    lon0, lat0 = MINLON, MINLAT
    span_x = (MAXLON - MINLON) * kx; span_y = (MAXLAT - MINLAT) * ky
    nx = int(math.ceil(span_x / CELL)); ny = int(math.ceil(span_y / CELL))
    H = np.zeros((nx, ny), np.float32)
    for (lat, lon), h in zip(polys, heights):
        x = (lon - lon0) * kx; y = (lat - lat0) * ky
        i0 = max(int(x.min() // CELL), 0); i1 = min(int(x.max() // CELL), nx - 1)
        j0 = max(int(y.min() // CELL), 0); j1 = min(int(y.max() // CELL), ny - 1)
        if i1 < i0 or j1 < j0:
            continue
        ii = np.arange(i0, i1 + 1); jj = np.arange(j0, j1 + 1)
        XX, YY = np.meshgrid((ii + 0.5) * CELL, (jj + 0.5) * CELL, indexing='ij')
        pts = np.column_stack([XX.ravel(), YY.ravel()])
        inside = Path(np.column_stack([x, y])).contains_points(pts).reshape(XX.shape)
        if inside.any():
            sub = H[i0:i1 + 1, j0:j1 + 1]
            np.maximum(sub, inside * np.float32(h), out=sub)
        else:  # footprint smaller than a cell: stamp its centroid cell
            ci = int(np.clip(x.mean() // CELL, 0, nx - 1))
            cj = int(np.clip(y.mean() // CELL, 0, ny - 1))
            H[ci, cj] = max(H[ci, cj], np.float32(h))
    return H, CELL, span_x, span_y, (kx, ky, lat_c)

def place_gs(H, cell):
    """Ground station on a central mid-rise rooftop: nearest-to-center cell whose
    building height is in [30,50] m; mast top clipped to [35,50] m."""
    nx, ny = H.shape; ci, cj = nx // 2, ny // 2
    roof = np.argwhere((H >= 30.0) & (H <= 50.0))
    if len(roof):
        gi, gj = roof[np.argmin((roof[:, 0] - ci) ** 2 + (roof[:, 1] - cj) ** 2)]
        gh = float(H[gi, gj])
    else:
        gi, gj, gh = ci, cj, float(H[ci, cj])
    gx = (gi + 0.5) * cell; gy = (gj + 0.5) * cell
    gz = float(np.clip((gh if gh >= 30 else 42.0) + 3.0, 35.0, 50.0))
    return np.array([gx, gy, gz]), (int(gi), int(gj), gh)

# ------------------------- trace generation (mirrors U.gen_traces) ------------
def gen_traces_osm(H, cell, span_x, span_y, gs, N=N_UAV, T=T_SLOTS, seed=GEO_SEED):
    """Straight patrol trajectories at 80-120 m wrapping in the district, with an
    exact 3D LoS ray-march to the GS over the REAL height grid via
    urban_blockage.los_blocked (reused unchanged; step count adapted to the link
    length so the march resolves the ~6 m grid)."""
    rng = np.random.default_rng(seed)
    los = np.zeros((T, N), bool); dist = np.zeros((T, N))
    for n in range(N):
        h = rng.uniform(80, 120)
        p0 = np.array([rng.uniform(0, span_x), rng.uniform(0, span_y), h])
        ang = rng.uniform(0, 2 * np.pi); v = rng.uniform(6, 14) * U.TAU     # m/slot
        vel = np.array([np.cos(ang), np.sin(ang), 0.0]) * v
        for t in range(T):
            p = p0 + vel * t
            p[0] = p[0] % span_x; p[1] = p[1] % span_y                      # wrap in district
            seg = float(np.linalg.norm(p - gs))
            steps = int(np.clip(seg / (cell * 0.7), 80, 400))               # resolve the grid
            blk = U.los_blocked(p[0], p[1], p[2], gs[0], gs[1], gs[2], H, cell, span_x, steps=steps)
            los[t, n] = (not blk); dist[t, n] = seg
    return los, dist

def autocorr_avg(los):
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.nanmean([U.autocorr(los[:, n]) for n in range(los.shape[1])], axis=0)

def load_traces():
    """Build (or load cached) the OSM LoS/distance traces so downstream figure
    scripts reproduce fast/offline without re-ray-marching."""
    tc = os.path.join(DATA, f"osm_traces_{SLUG}.npz")
    if os.path.exists(tc):
        z = np.load(tc); return z['los'], z['dist']
    data, _ = fetch_or_load(); polys, heights, _ = parse_buildings(data)
    H, cell, sx, sy, _ = build_grid(polys, heights); gs, _ = place_gs(H, cell)
    los, dist = gen_traces_osm(H, cell, sx, sy, gs)
    np.savez_compressed(tc, los=los, dist=dist); return los, dist

def eval_sched(los, dist, seeds=tuple(RUN_SEEDS)):
    """(wAoI, power/UAV) for Belief-Whittle and Memoryless via the UNCHANGED
    paper code (U.run_on_trace / U.belief_whittle / U.memoryless), averaged over
    run seeds."""
    return {name: tuple(np.mean([U.run_on_trace(los, dist, pol, seed=s) for s in seeds], axis=0))
            for name, pol in [("Belief-Whittle (ours)", U.belief_whittle),
                              ("Memoryless", U.memoryless)]}

def dominance(res):
    bw, ml = res["Belief-Whittle (ours)"], res["Memoryless"]
    dAoI = 100 * (ml[0] - bw[0]) / ml[0]
    dPow = 100 * (ml[1] - bw[1]) / ml[1]
    dom = (bw[0] <= ml[0] and bw[1] <= ml[1]) and (bw[0] < ml[0] or bw[1] < ml[1])
    return bw, ml, dAoI, dPow, dom

# ------------------------- main ----------------------------------------------
if __name__ == "__main__":
    print("=" * 74)
    print(f"REAL-DATA urban anchor (OSM footprints):  {CITY}")
    print(f"bbox (S,W,N,E) = ({MINLAT}, {MINLON}, {MAXLAT}, {MAXLON})")
    print("Building data (C) OpenStreetMap contributors, ODbL")
    print("=" * 74)

    data, origin = fetch_or_load()
    polys, heights, src = parse_buildings(data)
    nb = len(polys)
    if nb == 0:
        raise SystemExit("[FATAL] no building ways parsed; check bbox / cache.")
    ntag = src['tagged']; ninf = src['levels'] + src['default']
    print(f"[buildings] {nb} footprints  (source: {origin})")
    print(f"[heights]  tagged(OSM 'height')={ntag} ({100*ntag/nb:.1f}%) | "
          f"inferred(levelsx3)={src['levels']} ({100*src['levels']/nb:.1f}%) | "
          f"inferred(default 10 m)={src['default']} ({100*src['default']/nb:.1f}%)")
    print(f"[heights]  => {100*ntag/nb:.1f}% real-tagged, {100*ninf/nb:.1f}% inferred | "
          f"median={np.median(heights):.1f} m  p90={np.percentile(heights,90):.1f} m  "
          f"max={heights.max():.1f} m")

    H, cell, span_x, span_y, proj = build_grid(polys, heights)
    print(f"[grid]     {H.shape[0]}x{H.shape[1]} cells @ {cell:.0f} m  |  "
          f"span {span_x:.0f} x {span_y:.0f} m  |  built-up {(H>0).mean()*100:.1f}% of cells")
    gs, (gi, gj, gh) = place_gs(H, cell)
    print(f"[GS]       central rooftop @ ({gs[0]:.0f}, {gs[1]:.0f}) m, "
          f"local roof {gh:.0f} m, mast top z={gs[2]:.1f} m")

    print(f"[traces]   ray-marching UAV->GS LoS over REAL height grid "
          f"(N={N_UAV}, T={T_SLOTS}, geo seed={GEO_SEED})...")
    los, dist = gen_traces_osm(H, cell, span_x, span_y, gs)
    losfrac = float(los.mean())
    print(f"[LoS]      overall LoS fraction = {losfrac:.3f}  (NLoS = {1-losfrac:.3f})")
    ac = autocorr_avg(los)
    print(f"[autocorr] per-UAV blockage autocorr  lag1={ac[0]:.3f}  lag5={ac[4]:.3f}  "
          f"lag10={ac[9]:.3f}   (strongly correlated -> non-Markov)")

    # ---- scheduler comparison: SAME code as the paper (imported unchanged) ----
    res = eval_sched(los, dist)
    nseed = len(tuple(RUN_SEEDS))
    for name in ("Belief-Whittle (ours)", "Memoryless"):
        wa, en = res[name]
        print(f"[sched]    {name:22s} wAoI={wa:8.1f}   power/UAV={en:.3f} mW   (avg over {nseed} run seeds)")
    bw, ml, dAoI, dPow, dominate = dominance(res)
    verdict = "Pareto-DOMINATES" if dominate else "Pareto-TRADES (does NOT dominate)"
    print(f"[RESULT]   Belief-Whittle AoI gain = {dAoI:+.1f}%   power saving = {dPow:+.1f}% "
          f"->  Belief-Whittle {verdict} Memoryless")

    # ---- robustness: dominance across independent trajectory realizations -----
    print("[robust]   stability across independent geometry (trajectory) seeds:")
    for gseed in (0, 1, 2):
        if gseed == GEO_SEED:
            lo, di, lf = los, dist, losfrac
        else:
            lo, di = gen_traces_osm(H, cell, span_x, span_y, gs, seed=gseed); lf = float(lo.mean())
        _, _, dA, dP, dm = dominance(eval_sched(lo, di))
        print(f"             geo seed {gseed}: LoS={lf:.3f}  dAoI={dA:+5.1f}%  power={dP:+5.1f}%  dominates={dm}")

    # Seed-level scheduler values for error bars and open-source audit.
    sched_samples = {}
    sched_rows = []
    for name, policy in [('FreshSky-B', U.belief_whittle),
                         ('Wang 2026', U.wang26),
                         ('Memoryless', U.memoryless)]:
        values = np.asarray([U.run_on_trace(los, dist, policy, seed=s)
                             for s in RUN_SEEDS], float)
        sched_samples[name] = values
        for seed, (wa, en) in zip(RUN_SEEDS, values):
            sched_rows.append((name, seed, wa, en))
    import csv
    with open(os.path.join(HERE, 'results', 'fig5_osm_seeds.csv'),
              'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['method', 'seed', 'weighted_AoI', 'power_per_UAV_mW'])
        writer.writerows(sched_rows)

    # ---- final-size, two-panel real-geometry evidence -------------------------
    fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.55))
    L = len(ac)
    ax[0].plot(range(1, L + 1), ac, '-o', ms=2.2, lw=.9, color=GREEN)
    ax[0].axhline(0, ls=':', color='0.6')
    ax[0].set_xlabel("lag (slots)"); ax[0].set_ylabel("blockage autocorr.")
    ax[0].set_title("(a) Real OSM blockage")
    ax[0].text(0.96, 0.93,
               f"{CITY_SHORT}\n{nb} OSM buildings\n{100*ntag/nb:.0f}% heights tagged",
               transform=ax[0].transAxes, ha='right', va='top', fontsize=5.2)
    styles = {'FreshSky-B': ('o', GREEN), 'Wang 2026': ('D', ORANGE),
              'Memoryless': ('s', BLUE)}
    for name, values in sched_samples.items():
        marker, color = styles[name]
        ax[1].scatter(values[:, 1], values[:, 0], marker=marker, s=8,
                      color=color, alpha=.28, edgecolor='none', zorder=2)
        mean = values.mean(axis=0); sd = values.std(axis=0, ddof=1)
        ax[1].errorbar(mean[1], mean[0], xerr=sd[1], yerr=sd[0], fmt=marker,
                       ms=4.6, color=color, mec=BLACK, mew=.4, capsize=1.6,
                       elinewidth=.6, label=name, zorder=4)
    ax[1].set_xlabel("power per UAV (mW)"); ax[1].set_ylabel("weighted AoI")
    ax[1].set_title("(b) Pareto on OSM geometry"); ax[1].legend(fontsize=8, loc='upper left')
    ax[1].margins(0.18)                                   # breathing room around the 2 markers
    ax[1].legend(fontsize=5.3, loc='upper left', handletextpad=.2,
                 borderaxespad=.2)
    try:  # inset: the actual rasterized district footprint (north up), lower-right (empty) quadrant
        iax = ax[1].inset_axes([0.60, 0.09, 0.37, 0.37])
        vmax = float(np.percentile(H[H > 0], 95)) if (H > 0).any() else 1.0
        iax.imshow(H.T, origin='lower', cmap='Greys', aspect='auto', vmax=vmax)
        iax.set_xticks([]); iax.set_yticks([])
        iax.set_title(CITY_SHORT, fontsize=5.2)
    except Exception as ex:
        print("[fig] inset skipped:", ex)
    fig.tight_layout()
    out = os.path.join(FIG, "e13_osm.png")
    fig.savefig(os.path.join(FIG, 'e13_osm.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"[fig]      saved {os.path.relpath(out, HERE)}")
    print("=" * 74)
    print("Building data (C) OpenStreetMap contributors, ODbL. "
          "Raw response cached for offline reproduction.")
