# Academic Figure Skill Asset Confirmation Table
# Fig. 2 line panels: LineTrend -> PARAM-INHERIT (asset data incompatible).
# Fig. 2 ablation: BarAblation -> PARAM-INHERIT (asset data incompatible).
# Fig. 2 scatter: BarComparison -> CROSS-TYPE PARAM-INHERIT.
# Fig. 3 lines/bars: LineTrend and BarComparison -> PARAM-INHERIT.
# Inherited: typography, palette, open axes, line/marker redundancy, error bars.
# Not inherited: any production-asset values, labels, or scientific claims.
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Academic Figure Skill Typography Baseline - final-size publication settings.
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'],
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 8,
    'figure.titlesize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'legend.frameon': False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette.
CATEGORICAL = ['#2166AC', '#B2182B', '#1B7837', '#F1A340', '#762A83', '#666666']
CATEGORICAL_EXTENDED = [
    '#2166AC', '#B2182B', '#1B7837', '#F1A340', '#762A83', '#666666',
    '#4393C3', '#D6604D', '#5AAE61', '#B35806', '#9970AB', '#999999',
]
DIVERGING = ['#2166AC', '#F7F7F7', '#B2182B']
SEQUENTIAL = ['#F7FBFF', '#6BAED6', '#08306B']
ACCENT_RED = '#B2182B'
GREY = '#999999'
BLACK = '#222222'

# Academic Figure Skill Export Baseline - vector PDF plus 300 dpi PNG.
mpl.rcParams.update({
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.bbox': 'tight',
    'savefig.dpi': 300,
})

def save_cns_figure(fig, filename):
    fig.savefig(filename + '.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(filename + '.png', bbox_inches='tight', dpi=300)

import freshsky_sim as F
import belief_whittle as BW
import ablation as AB
import conv_fig as CV
import robustness_shift as RS

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figs')
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
SEEDS = (0, 1, 2, 3, 4)
BLUE, RED, GREEN, ORANGE, PURPLE, GREY2 = CATEGORICAL
METHOD = {
    'FreshSky': (GREEN, 'o', '-'),
    'AoI-Energy': (BLUE, 's', '--'),
    'Lyap-DPP': (PURPLE, '^', ':'),
    'Max-Weight': (RED, 'D', '-.'),
    'Fixed-Price': (ORANGE, 'v', '--'),
}

def panel(axis, letter):
    axis.text(0.015, 0.985, '(' + letter + ')', transform=axis.transAxes,
              fontsize=8, fontweight='bold', va='top', ha='left', zorder=20,
              bbox=dict(facecolor='white', edgecolor='none', alpha=.82, pad=.15))
    axis.tick_params(length=2.5, pad=1.5)
    axis.grid(axis='y', color='#D9D9D9', lw=0.45, alpha=0.65, zorder=0)

def mean_sd(values):
    values = np.asarray(values, float)
    return values.mean(), values.std(ddof=1)

def line(axis, x, y, label, key, yerr=None, markevery=None):
    color, marker, ls = METHOD[key]
    if yerr is None:
        axis.plot(x, y, color=color, marker=marker, ls=ls, lw=1.1,
                  ms=2.8, markevery=markevery, label=label)
    else:
        axis.errorbar(x, y, yerr=yerr, color=color, marker=marker, ls=ls,
                      lw=1.0, ms=2.8, capsize=1.6, elinewidth=0.6, label=label)

def save_rows(name, rows):
    pd.DataFrame(rows).to_csv(os.path.join(RES, name), index=False)

def build_main_grid():
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 2.72))
    ax = axes.ravel()
    base = F.Cfg().derived()

    data = pd.read_csv(os.path.join(RES, 'vsweep_hi.csv'))
    ax[0].set_xscale('log')
    ax[0].errorbar(data.V, data.wAoI, yerr=data.wAoI_std, color=BLUE,
                   marker='o', ms=2.8, lw=1.0, capsize=1.5, elinewidth=0.6)
    ax[0].set_xlabel('control weight $V$')
    ax[0].set_ylabel('weighted AoI', color=BLUE)
    ax[0].tick_params(axis='y', labelcolor=BLUE)
    twin = ax[0].twinx()
    twin.plot(data.V, data.Qbacklog, color=RED, marker='s', ms=2.6, lw=1.0, ls='--')
    twin.set_ylabel('energy backlog', color=RED, fontsize=7)
    twin.tick_params(axis='y', labelcolor=RED, labelsize=6.2, length=2.5, pad=1)
    twin.spines['top'].set_visible(False)
    panel(ax[0], 'a')

    cfg = BW.clone(BW.Cfg(), V=30.0, pbar=0.22)
    variants = [('Full', AB.full), ('No value', AB.no_value),
                ('No belief', AB.no_belief), ('No queue', AB.no_queue)]
    rows, means, sds, feasible = [], [], [], []
    for label, policy in variants:
        values, power = [], []
        for seed in SEEDS:
            wa, pw = BW.run(policy, cfg, seeds=(seed,))
            values.append(wa); power.append(pw)
            rows.append(dict(variant=label, seed=seed, weighted_AoI=wa,
                             power_per_UAV_mW=pw))
        mean, sd = mean_sd(values)
        means.append(mean); sds.append(sd); feasible.append(max(power) <= cfg.pbar * 1.02)
    bars = ax[1].bar(range(4), means, yerr=sds,
                     color=[GREEN if ok else RED for ok in feasible],
                     edgecolor=BLACK, lw=0.45, capsize=1.8, zorder=2)
    for index, bar in enumerate(bars):
        if not feasible[index]:
            bar.set_hatch('////')
        values = [r['weighted_AoI'] for r in rows if r['variant'] == variants[index][0]]
        ax[1].scatter(index + np.linspace(-0.10, 0.10, len(values)), values,
                      s=7, facecolor='white', edgecolor=BLACK, lw=0.4, zorder=3)
    ax[1].set_xticks(range(4), ['Full', 'No\nvalue', 'No\nbelief', 'No\nqueue'])
    ax[1].set_ylabel('weighted AoI')
    panel(ax[1], 'b')
    save_rows('fig2_ablation_seeds.csv', rows)

    series = [('FreshSky', F.FreshSkyDPP, 'FreshSky'),
              ('AoI-Energy 2025', F.AoIEnergyOnline, 'AoI-Energy'),
              ('Lyap-DPP', F.DPP_NoValue, 'Lyap-DPP'),
              ('Max-Weight', F.MaxWeight, 'Max-Weight')]
    for label, policy, key in series:
        x, y, sd = CV.running_wAoI(policy, F.clone(base, V=200.0), return_stats=True)
        line(ax[2], x, y, label, key, markevery=520)
        if key == 'FreshSky':
            ax[2].fill_between(x, y - sd, y + sd, color=GREEN, alpha=0.14, lw=0)
    ax[2].set_xlabel('slot $t$')
    ax[2].set_ylabel('running weighted AoI')
    ax[2].legend(fontsize=5.5, ncol=2, loc='upper right', handlelength=1.7,
                 columnspacing=0.7, borderaxespad=0.2)
    panel(ax[2], 'c')

    cfg = BW.clone(BW.Cfg(), pbar=0.4)
    methods = [('Genie', BW.belief_whittle, True, GREY2, '*'),
               ('FreshSky-B', BW.belief_whittle, False, GREEN, 'o'),
               ('Belief-DPP 2026', BW.BeliefDPP26, False, PURPLE, 'X'),
               ('Memoryless', BW.memoryless, False, BLUE, 's')]
    rows = []
    for label, policy, genie, color, marker in methods:
        xs, ys = [], []
        for seed in SEEDS:
            with np.errstate(invalid='ignore', divide='ignore'):
                result = BW.run(policy, cfg, seeds=(seed,), genie=genie, details=True)
            xs.append(result['max_power']); ys.append(result['wAoI'])
            rows.append(dict(method=label, seed=seed, weighted_AoI=result['wAoI'],
                             max_power_per_UAV_mW=result['max_power']))
        xm, xsd = mean_sd(xs); ym, ysd = mean_sd(ys)
        ax[3].scatter(xs, ys, s=8, color=color, alpha=0.25, edgecolor='none')
        ax[3].errorbar(xm, ym, xerr=xsd, yerr=ysd, fmt=marker, color=color,
                       mec=BLACK, mew=0.4, ms=4.8, capsize=1.6,
                       elinewidth=0.6, label=label, zorder=4)
    ax[3].axvline(cfg.pbar, color=BLACK, ls='--', lw=0.7)
    ax[3].set_xlabel('maximum per-UAV power (mW)')
    ax[3].set_ylabel('weighted AoI')
    ax[3].legend(fontsize=5.3, loc='best', handletextpad=0.2, borderaxespad=0.2)
    panel(ax[3], 'd')
    save_rows('fig2_belief_seeds.csv', rows)

    dwell = [1.5, 2, 3, 5, 8, 14, 20]
    rows, means, sds = [], [], []
    for value in dwell:
        gains = []
        for seed in SEEDS:
            cfg = BW.clone(BW.Cfg(), mean_dwell=value)
            belief, _ = BW.run(BW.belief_whittle, cfg, seeds=(seed,))
            memory, _ = BW.run(BW.memoryless, cfg, seeds=(seed,))
            gain = 100.0 * (memory - belief) / memory
            gains.append(gain)
            rows.append(dict(mean_NLoS_dwell=value, seed=seed, belief_gain_percent=gain))
        mean, sd = mean_sd(gains); means.append(mean); sds.append(sd)
    ax[4].errorbar(dwell, means, yerr=sds, color=GREEN, marker='o', ms=2.8,
                   lw=1.0, capsize=1.6, elinewidth=0.6)
    ax[4].axhline(0, color=GREY2, ls=':', lw=0.7)
    ax[4].set_xlabel('mean NLoS dwell (slots)')
    ax[4].set_ylabel('belief gain (percent)')
    panel(ax[4], 'e')
    save_rows('fig2_correlation_seeds.csv', rows)

    budget, shift_slot, horizon = base.pbar, 1500, 3000
    for value in [0.02, 0.03, 0.05, 0.07, 0.10]:
        F.FixedPriceThr.LAM = value
        _, power = RS.run_shift(F.FixedPriceThr, F.clone(base, V=200.0),
                                horizon + 10, shift_slot, {})
        if power.mean() <= budget:
            break
    def excess(power):
        cumulative = np.cumsum(power[shift_slot:] - budget)
        return max(0.0, float(cumulative.max()))
    severity = [0, 1, 2, 3, 4]
    methods = [('FreshSky', F.FreshSkyDPP, 'FreshSky'),
               ('AoI-Energy 2025', F.AoIEnergyOnline, 'AoI-Energy'),
               ('Fixed-Price', F.FixedPriceThr, 'Fixed-Price')]
    rows = []
    for label, policy, key in methods:
        means, sds = [], []
        for value in severity:
            shift = dict(p_ihigh=0.6, I_high=2e-8 * value, p_event=0.5,
                         mean_sojourn_ev=50.0) if value else {}
            samples = []
            for seed in SEEDS:
                _, power = RS.run_shift(policy, F.clone(base, V=200.0, seed=seed),
                                        shift_slot, horizon, shift)
                sample = excess(power); samples.append(sample)
                rows.append(dict(method=label, severity=value, seed=seed,
                                 peak_cumulative_excess=sample))
            mean, sd = mean_sd(samples)
            means.append(max(mean, 1.0)); sds.append(sd)
        errors = np.minimum(sds, 0.8 * np.asarray(means))
        line(ax[5], severity, means, label, key, yerr=errors)
    ax[5].set_yscale('log')
    ax[5].set_xticks(severity)
    ax[5].set_xlabel('shift severity')
    ax[5].set_ylabel('peak cumulative excess')
    ax[5].legend(fontsize=5.3, loc='upper left', handlelength=1.7,
                 handletextpad=0.3, borderaxespad=0.2)
    panel(ax[5], 'f')
    save_rows('fig2_shift_seeds.csv', rows)

    fig.subplots_adjust(left=0.068, right=0.955, bottom=0.17, top=0.95,
                        wspace=0.52, hspace=0.67)
    save_cns_figure(fig, os.path.join(FIG, 'fig_grid_main'))
    plt.close(fig)
    print('saved figs/fig_grid_main.{pdf,png}')

def build_robust_grid():
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 2.78))
    ax = axes.ravel()
    base = F.Cfg().derived()
    budget, shift_slot, horizon = base.pbar, 1500, 3000

    def moving_average(values, window=100):
        cumulative = np.cumsum(np.insert(values, 0, 0.0))
        return (cumulative[window:] - cumulative[:-window]) / window

    for value in [0.02, 0.03, 0.05, 0.07, 0.10]:
        F.FixedPriceThr.LAM = value
        _, power = RS.run_shift(F.FixedPriceThr, F.clone(base, V=200.0),
                                horizon + 10, shift_slot, {})
        if power.mean() <= budget:
            break
    shift = dict(p_ihigh=0.6, I_high=4e-8, p_event=0.5, mean_sojourn_ev=50.0)
    methods = [('FreshSky', F.FreshSkyDPP, 'FreshSky'),
               ('AoI-Energy 2025', F.AoIEnergyOnline, 'AoI-Energy'),
               ('Fixed-Price', F.FixedPriceThr, 'Fixed-Price')]
    rows = []
    for label, policy, key in methods:
        runs = [RS.run_shift(policy, F.clone(base, V=200.0, seed=seed),
                             shift_slot, horizon, shift) for seed in SEEDS]
        queue = np.asarray([item[0] for item in runs])
        power = np.asarray([item[1] for item in runs])
        color, marker, ls = METHOD[key]
        ax[0].plot(np.arange(99, horizon), moving_average(power.mean(axis=0)),
                   color=color, ls=ls, lw=1.05, label=label)
        ax[1].plot(np.arange(horizon), queue.mean(axis=0),
                   color=color, ls=ls, lw=1.05, label=label)
        for seed, q, p in zip(SEEDS, queue, power):
            rows.append(dict(method=label, seed=seed,
                             post_shift_power_mW=p[shift_slot + 500:].mean(),
                             final_energy_backlog=q[-1]))
    ax[0].axhline(budget, color=BLACK, ls=':', lw=0.7)
    ax[0].axvline(shift_slot, color=GREY2, ls='--', lw=0.7)
    ax[0].set_xlabel('slot $t$')
    ax[0].set_ylabel('per-UAV power (mW)')
    ax[0].legend(fontsize=5.3, loc='upper left', handlelength=1.7,
                 handletextpad=0.3, borderaxespad=0.2)
    panel(ax[0], 'a')
    ax[1].axvline(shift_slot, color=GREY2, ls='--', lw=0.7)
    ax[1].set_xlabel('slot $t$')
    ax[1].set_ylabel('total energy backlog')
    panel(ax[1], 'b')
    save_rows('fig3_shift_trajectories.csv', rows)

    def sweep(policy, parameter, values):
        means, sds, rows = [], [], []
        for value in values:
            samples = []
            for seed in SEEDS:
                kwargs = {parameter: value}
                if parameter == 'N':
                    kwargs['M'] = max(2, value // 4)
                cfg = F.clone(base, V=200.0, seed=seed, **kwargs)
                result = F.run_episode(policy, cfg)
                samples.append(result['wAoI'])
                rows.append(dict(parameter=parameter, value=value,
                                 policy=policy.name, seed=seed,
                                 weighted_AoI=result['wAoI']))
            mean, sd = mean_sd(samples); means.append(mean); sds.append(sd)
        return np.asarray(means), np.asarray(sds), rows

    specs = [('mean_sojourn_N', [2, 4, 6, 8, 12, 16], 'mean NLoS dwell (slots)'),
             ('p_ihigh', [0.1, 0.2, 0.3, 0.4, 0.5], 'interference probability'),
             ('N', [6, 9, 12, 16, 20, 32, 50], 'number of UAVs $N$')]
    methods = [('FreshSky', F.FreshSkyDPP, 'FreshSky'),
               ('Lyap-DPP', F.DPP_NoValue, 'Lyap-DPP'),
               ('Max-Weight', F.MaxWeight, 'Max-Weight')]
    all_rows = []
    for axis, spec in zip(ax[2:5], specs):
        parameter, values, xlabel = spec
        for label, policy, key in methods:
            means, sds, rows = sweep(policy, parameter, values)
            all_rows.extend(rows)
            line(axis, values, means, label, key, yerr=sds)
        axis.set_xlabel(xlabel)
        axis.set_ylabel('weighted AoI')
        axis.legend(fontsize=5.2, loc='best', handlelength=1.7,
                    handletextpad=0.3, borderaxespad=0.2)
    for letter, axis in zip('cde', ax[2:5]):
        panel(axis, letter)
    save_rows('fig3_sweep_seeds.csv', all_rows)

    latency_methods = [('FreshSky', F.FreshSkyDPP),
                       ('Lyap-DPP', F.DPP_NoValue),
                       ('Max-Weight', F.MaxWeight),
                       ('Max-Age', F.MaxAgeFirst),
                       ('Opportun.', F.Opportunistic)]
    rows, means, sds, feasible = [], [], [], []
    for label, policy in latency_methods:
        values, flags = [], []
        for seed in SEEDS:
            result = F.run_episode(policy, F.clone(base, V=200.0, seed=seed))
            values.append(result['det_latency']); flags.append(result['energy_ok'])
            rows.append(dict(method=label, seed=seed,
                             detection_latency=result['det_latency'],
                             energy_feasible=result['energy_ok']))
        mean, sd = mean_sd(values)
        means.append(mean); sds.append(sd); feasible.append(all(flags))
    bars = ax[5].bar(range(5), means, yerr=sds,
                     color=[GREEN if ok else RED for ok in feasible],
                     edgecolor=BLACK, lw=0.45, capsize=1.8, zorder=2)
    for index, bar in enumerate(bars):
        if not feasible[index]:
            bar.set_hatch('////')
        values = [r['detection_latency'] for r in rows
                  if r['method'] == latency_methods[index][0]]
        ax[5].scatter(index + np.linspace(-0.09, 0.09, len(values)), values,
                      s=6.5, facecolor='white', edgecolor=BLACK, lw=0.4, zorder=3)
    ax[5].set_xticks(range(5), ['FreshSky', 'Lyap', 'Max-W', 'Max-A', 'Opport.'],
                     rotation=24, ha='right')
    ax[5].set_ylabel('detection latency (slots)')
    panel(ax[5], 'f')
    save_rows('fig3_latency_seeds.csv', rows)

    fig.subplots_adjust(left=0.068, right=0.985, bottom=0.18, top=0.95,
                        wspace=0.40, hspace=0.72)
    save_cns_figure(fig, os.path.join(FIG, 'fig_grid_robust'))
    plt.close(fig)
    print('saved figs/fig_grid_robust.{pdf,png}')

def main():
    build_main_grid()
    build_robust_grid()

if __name__ == '__main__':
    main()
