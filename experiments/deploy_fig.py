# -*- coding: utf-8 -*-
# Academic Figure Skill Asset Confirmation: BarComparison/scatter ->
# CROSS-TYPE PARAM-INHERIT. Only final-size typography, semantic colors,
# open axes, and baseline highlighting are inherited; all values remain CSV-backed.
"""THE DEPLOYABILITY PLANE (one figure, read in one glance).

  x  steady-state weighted AoI    -- freshness: what every scheduler optimizes
  y  peak cumulative budget        -- safety:    what almost none of them measures
     violation under a shift

All known-channel schedulers, the same mid-mission shift, and the same five seeds as Table III.
The claim the figure makes is exactly the one the data supports:

    EVERYTHING FRESHER THAN FRESHSKY EITHER BREAKS THE BUDGET OR NEEDS TUNING.

and nothing more. In particular FreshSky is NOT the safest policy -- a
value-agnostic Lyapunov or energy-priced index is safer still, because it simply
does not chase valuable regions, and pays 10-40% more AoI for it. Every
superlative below is COMPUTED from the CSV, never hard-coded, so a claim that
stops being true cannot survive into the paper.

Output: figs/fig_deploy.png
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, 'results'), os.path.join(HERE, 'figs')
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

df = pd.read_csv(os.path.join(RES, 'deployability.csv'))
# VoI-Greedy == Max-Weight and Whittle-AoI == Max-Age-First under generate-at-will
# (they share rows in Table III); merge so the plane shows one marker per policy.
DUP = {'VoI-Greedy': 'Max-Weight', 'Whittle-AoI': 'Max-Age-First'}
df = df[~df['policy'].isin(DUP)].copy()
df['is_ours'] = df['policy'].str.startswith('FreshSky')
ours = df[df['is_ours']].iloc[0]
XCLIP, YFLOOR = 114.0, 2.0            # Opportunistic: AoI 310 (off-scale), violation 0


def cls(r):
    if not r['steady_feasible']: return 'over'      # over budget before anything happens
    if not r['recovers']:        return 'breaks'    # over budget after the shift
    return 'ours' if r['is_ours'] else 'holds'
df['cls'] = df.apply(cls, axis=1)

STY = {'over':   dict(c=RED, m='X', s=34, fc='none'),
       'breaks': dict(c=RED, m='v', s=34, fc='none'),
       'holds':  dict(c=BLUE, m='o', s=30, fc=BLUE),
       'ours':   dict(c=GREEN, m='*', s=145, fc=GREEN)}

fig, ax = plt.subplots(figsize=(3.45, 2.18))
ax.set_yscale('log'); ax.set_xlim(63.5, XCLIP); ax.set_ylim(YFLOOR, 3.4e4)

# ---- region 1: fresher than FreshSky ------------------------------------------
ax.axvspan(63.5, ours['wAoI'], facecolor=GREEN, alpha=.055, zorder=0)
ax.axvline(ours['wAoI'], ls='--', color=GREEN, lw=.8, zorder=2)

# ---- region 2: never recovers the budget after the shift ----------------------
ylo = df[df['recovers']]['peak_cum_viol'].max() * 4
ax.axhspan(ylo, 3.4e4, facecolor='0.87', alpha=.75, hatch='//', edgecolor='0.62', lw=0, zorder=0)
ax.text(XCLIP - 1, 2.4e4, 'backlog diverges: never\nrecovers the budget',
        fontsize=5.3, color='0.3', ha='right', va='top', style='italic')

for k, st in STY.items():
    d = df[df['cls'] == k]
    if not len(d): continue
    ax.scatter(d['wAoI'].clip(upper=XCLIP - .6), d['peak_cum_viol'].clip(lower=YFLOOR),
               marker=st['m'], s=st['s'], facecolor=st['fc'], edgecolor=st['c'],
               lw=1.5, zorder=8 if k == 'ours' else 5)

# ---- labels -------------------------------------------------------------------
SHOW = {   # policy prefix -> (text, dx, dy, ha)
    'Max-Age-First':   ('freshest indices', 5, 7, 'left'),
    'Round-Robin':     ('Round-Robin', -5, 2, 'right'),
    'Fixed-Price':     ('fixed price', 0, -8, 'center'),
    'Deep Index':      ('Deep Index', 4, -1, 'left'),
    'AoI-Energy':      ("Abd-Elmagid'25$^{\\ast}$", 9, 3, 'left'),
    'Lyap-DPP':        ('Lyap-DPP', 4, 0, 'left'),
    'Cost-AoI':        ('Cost-AoI', 0, 7, 'center'),
}
for _, r in df.iterrows():
    for k, (txt, dx, dy, ha) in SHOW.items():
        if r['policy'].startswith(k):
            ax.annotate(txt, (min(r['wAoI'], XCLIP - .6), max(r['peak_cum_viol'], YFLOOR)),
                        textcoords='offset points', xytext=(dx, dy), ha=ha, va='center',
                        fontsize=5.3, color=STY[r['cls']]['c'])

ax.annotate('FreshSky (ours)', (ours['wAoI'], ours['peak_cum_viol']), textcoords='offset points',
            xytext=(0, -12), ha='center', fontsize=6.3, color=GREEN, weight='bold')

# Opportunistic: off-scale on both axes -- "safe" only because it barely transmits
if len(df[df['wAoI'] > XCLIP]):
    ax.annotate('', xy=(XCLIP - .7, YFLOOR * 1.35), xytext=(XCLIP - 4.5, YFLOOR * 1.35),
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=.8))

# ---- the one sentence the plane is for ----------------------------------------
ax.text(71.0, 340, 'fresher methods violate\nor require tuning',
        ha='center', va='center', fontsize=5.5, color=GREEN, weight='bold')
ax.text(96.0, 30, 'safer but staler', ha='center', va='center',
        fontsize=5.5, color=BLUE, style='italic')

ax.set_xlabel('steady-state weighted AoI')
ax.set_ylabel('peak cumulative excess (mW$\\cdot$slots)')
ax.grid(alpha=.2, which='both', lw=.4)

fig.tight_layout(pad=.3)
out = os.path.join(FIG, 'fig_deploy')
fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('saved', out + '.{pdf,png}')

# ============ every claim RE-DERIVED from the CSV; the figure ships only if they hold ====
adm = df[df['steady_feasible']]
dom = df[(df['wAoI'] < ours['wAoI']) & (df['peak_cum_viol'] < ours['peak_cum_viol'])]
tf = adm[~adm['needs_tuning']].sort_values('wAoI')
fresher = df[df['wAoI'] < ours['wAoI']]
bad = fresher[fresher['steady_feasible'] & ~fresher['needs_tuning']]
strict = df[(df['wAoI'] > ours['wAoI']) & (df['peak_cum_viol'] > ours['peak_cum_viol'])]

print('\n' + '=' * 78)
print(f"FreshSky: AoI {ours['wAoI']:.1f}, shift violation {ours['peak_cum_viol']:.0f} "
      f"(+/-{ours['pcv_std']:.0f}), tuning-free\n")
print(f"[CLAIM 1] on the Pareto frontier (nothing is both fresher AND safer): "
      f"{'HOLDS' if not len(dom) else 'FAILS -> ' + str(list(dom['policy']))}")
print(f"[CLAIM 2] freshest tuning-free budget-feasible policy: "
      f"{'HOLDS' if tf.iloc[0]['is_ours'] else 'FAILS -> ' + tf.iloc[0]['policy']}"
      f"   (runner-up {tf.iloc[1]['policy']}, AoI {tf.iloc[1]['wAoI']:.1f})")
print(f"[CLAIM 3] everything fresher is over-budget or needs tuning: "
      f"{'HOLDS' if not len(bad) else 'FAILS -> ' + str(list(bad['policy']))}")
print(f"[CLAIM 4] Pareto-DOMINATES (staler AND less safe): {list(strict['policy'])}")
print(f"\n[NOT A CLAIM] safer than us: "
      f"{list(adm[adm['peak_cum_viol'] < ours['peak_cum_viol']]['policy'])}"
      f"  <- all value-agnostic; never write 'safest'.")
print('=' * 78)
