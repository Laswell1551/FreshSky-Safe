# -*- coding: utf-8 -*-
"""Build DENSE multi-panel result figures for the paper. Reads the committed
CSVs for the main-table / money-plot / ablation panels and recomputes the fast
experiments (convergence, belief, correlation, real-OSM, shift severity) via the
UNCHANGED paper modules. Emits fig_grid_main.png (2x4) and fig_grid_robust.png
(2x3). Grayscale-safe (distinct markers/linestyles + hatching)."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); RES=os.path.join(HERE,'results'); FIG=os.path.join(HERE,'figs')
import freshsky_sim as F
import belief_whittle as BW
import conv_fig as CV
import robustness_shift as RS
import osm_urban as OU
plt.rcParams.update({'font.size':8})
def short(s): return s.replace(' (ours)','*').replace('Deep Index Policy (RL)','DeepIndex').replace(' [Abd-Elmagid\'25]','')

# ============================ GRID 1: core results (2x4) ============================
fig,axz=plt.subplots(2,4,figsize=(13.6,4.55)); ax=axz.ravel()

df=pd.read_csv(os.path.join(RES,'summary_default.csv'))
representative=['CW-Whittle', 'Max-Weight', 'AoI-Energy Online', 'FreshSky-DPP',
                'Tang-CMDP', 'Fixed-Price', 'Lyap-DPP', 'Deep Index', 'Opportunistic']
df=df[df['policy'].apply(lambda p:any(p.startswith(k) for k in representative))].sort_values('wAoI').reset_index(drop=True)
cols=['#2ca02c' if ok else '#d62728' for ok in df['energy_ok']]
bars=ax[0].bar(range(len(df)),df['wAoI'],color=cols,edgecolor='k',lw=.5)
for b,ok in zip(bars,df['energy_ok']):
    if not ok: b.set_hatch('////')
ax[0].set_xticks(range(len(df))); ax[0].set_xticklabels([short(p) for p in df['policy']],rotation=42,ha='right',fontsize=5.5)
ax[0].set_ylabel('weighted AoI'); ax[0].set_title('(a) AoI: feasible (green) vs budget-violating (hatched)')

mp=pd.read_csv(os.path.join(RES,'moneyplot.csv')); a=ax[1]; a.set_xscale('log')
a.plot(mp['V'],mp['wAoI'],'o-',color='#1f77b4'); a.set_xlabel('Lyapunov knob $V$')
a.set_ylabel('weighted AoI',color='#1f77b4'); a.tick_params(axis='y',labelcolor='#1f77b4')
a2=a.twinx(); a2.plot(mp['V'],mp['Qbacklog'],'s--',color='#d62728'); a2.set_ylabel('energy backlog',color='#d62728'); a2.tick_params(axis='y',labelcolor='#d62728')
a.set_title('(b) $[O(1/V),O(V)]$ tradeoff (Thm 2)')

ab=pd.read_csv(os.path.join(RES,'ablation.csv'))
cols=['#2ca02c' if f else '#d62728' for f in ab['feasible']]
bb=ax[2].bar(range(len(ab)),ab['wAoI'],color=cols,edgecolor='k',lw=.5)
for b,f in zip(bb,ab['feasible']):
    if not f: b.set_hatch('////')
labs=[str(v).replace('$','').replace('\\','').replace('-','−') for v in ab['variant']]
ax[2].set_xticks(range(len(ab))); ax[2].set_xticklabels(labs,rotation=22,ha='right',fontsize=6)
ax[2].set_ylabel('weighted AoI'); ax[2].set_title('(c) Ablation: each component binds')

base=F.Cfg().derived()
for name,P,ls,c in [('FreshSky',F.FreshSkyDPP,'-','#2ca02c'),("Abd-Elmagid'25",F.AoIEnergyOnline,'--','#1f77b4'),
                    ('Lyap-DPP',F.DPP_NoValue,':','#9467bd'),('Max-Weight (infeas.)',F.MaxWeight,'-.','#d62728')]:
    x,y=CV.running_wAoI(P,F.clone(base,V=200.0)); ax[3].plot(x,y,ls=ls,color=c,lw=1.3,label=name)
ax[3].set_xlabel('slot $t$'); ax[3].set_ylabel('running AoI'); ax[3].legend(fontsize=5.6); ax[3].set_title('(d) Convergence')

cfg=BW.Cfg(); cfg_tight=BW.clone(cfg,pbar=0.4)
for name,pol,g,m,c in [('Genie',BW.belief_whittle,True,'*','0.4'),('FreshSky-B*',BW.belief_whittle,False,'o','#2ca02c'),
                       ("Belief-DPP'26 (viol.)",BW.BeliefDPP26,False,'X','#9467bd'),('Memoryless',BW.memoryless,False,'s','#1f77b4')]:
    wA,en=BW.run(pol,cfg_tight,genie=g); ax[4].scatter(en,wA,marker=m,s=95,color=c,edgecolor='k',label=name,zorder=3)
ax[4].set_xlabel('power/UAV (mW)'); ax[4].set_ylabel('weighted AoI'); ax[4].legend(fontsize=6); ax[4].set_title('(e) Partial obs.: per-UAV budget') ; ax[4].margins(0.18)

dwell=[1.5,2,3,5,8,14,20]; gain=[]
for md in dwell:
    c=BW.clone(cfg,mean_dwell=md); bw,_=BW.run(BW.belief_whittle,c); ml,_=BW.run(BW.memoryless,c); gain.append(100*(ml-bw)/ml)
ax[5].plot(dwell,gain,'-o',color='#2ca02c'); ax[5].axhline(0,ls=':',color='0.6')
ax[5].set_xlabel('mean NLoS dwell = correlation'); ax[5].set_ylabel('belief gain (\\%)'); ax[5].set_title('(f) Gain grows with correlation')

los,dist=OU.load_traces(); res=OU.eval_sched(los,dist)
for name,(wa,en) in res.items():
    m,c=(('o','#2ca02c') if 'Belief' in name else ('s','#1f77b4')); ax[6].scatter(en,wa,marker=m,s=95,color=c,edgecolor='k',label=name.replace(' (ours)','*'),zorder=3)
waw,enw=np.mean([OU.U.run_on_trace(los,dist,OU.U.wang26,seed=s) for s in range(4)],axis=0)
ax[6].scatter(enw,waw,marker='D',s=80,color='#ff7f0e',edgecolor='k',label="Wang'26 POMDP",zorder=3)
ax[6].set_xlabel('power/UAV (mW)'); ax[6].set_ylabel('weighted AoI'); ax[6].legend(fontsize=5.6); ax[6].set_title('(g) Zero-shot to real OSM Manhattan'); ax[6].margins(0.2)

# shift severity sweep (recompute, 5 seeds, mean +/- std)
bud=base.pbar; ts=1500; T=3000
for lam in [0.02,0.03,0.05,0.07,0.10]:
    F.FixedPriceThr.LAM=lam; _,pa=RS.run_shift(F.FixedPriceThr,F.clone(base,V=200.0),T+10,ts,{})
    if pa.mean()<=bud: break
def pce(pp): d=np.cumsum(pp[ts:]-bud); return max(0.0,d.max())
sev=[0,1,2,3,4]; curves={'Fixed-Price':('--','#d62728',F.FixedPriceThr),"Abd-Elmagid'25":('-.','#9467bd',F.AoIEnergyOnline),'FreshSky*':('-','#2ca02c',F.FreshSkyDPP)}
for lab,(ls,c,cls) in curves.items():
    ys=[]; es=[]
    for s in sev:
        sh=dict(p_ihigh=0.6,I_high=2e-8*s,p_event=0.5,mean_sojourn_ev=50.0) if s>0 else {}
        vals=[pce(RS.run_shift(cls,F.clone(base,V=200.0,seed=sd),ts,T,sh)[1]) for sd in range(5)]
        ys.append(np.mean(vals)); es.append(np.std(vals))
    ys=np.maximum(ys,1.0)
    ax[7].errorbar(sev,ys,yerr=es,fmt=ls+'o',ms=4,color=c,label=lab,capsize=2,lw=1.2,elinewidth=.8)
ax[7].set_yscale('log')
ax[7].set_xlabel('shift severity (interf. $\\times$)'); ax[7].set_ylabel('peak cum. excess'); ax[7].legend(fontsize=5.8); ax[7].set_title('(h) Bounded violation under shift (Prop 4)')

fig.tight_layout(pad=0.6); fig.savefig(os.path.join(FIG,'fig_grid_main.png'),dpi=155,bbox_inches='tight'); plt.close(fig)
print('saved fig_grid_main.png')

# ============================ GRID 2: robustness & shift (2x3) ============================
fig,axz=plt.subplots(2,3,figsize=(11.0,4.45)); ax=axz.ravel()
def mavg(x,w=100): c=np.cumsum(np.insert(x,0,0)); return (c[w:]-c[:-w])/w
SEEDS5=(0,1,2,3,4)
def sweep_ms(P,mods,seeds=SEEDS5):
    """weighted-AoI mean and std over `seeds`, one point per config in `mods`."""
    rs=[F.run_multi(P,F.clone(base,V=200.0,**m),seeds=seeds) for m in mods]
    return np.array([r['wAoI'] for r in rs]), np.array([r['wAoI_std'] for r in rs])

# distribution shift: power + backlog (offline-tune the fixed price first)
for lam in [0.02,0.03,0.05,0.07,0.10]:
    F.FixedPriceThr.LAM=lam; _,pa=RS.run_shift(F.FixedPriceThr,F.clone(base,V=200.0),T+10,ts,{})
    if pa.mean()<=bud: break
sh=dict(p_ihigh=0.6,I_high=4e-8,p_event=0.5,mean_sojourn_ev=50.0)
runs={'Fixed-Price':('--','#d62728',F.FixedPriceThr),"Abd-Elmagid'25":('-.','#9467bd',F.AoIEnergyOnline),'FreshSky*':('-','#2ca02c',F.FreshSkyDPP)}
QP={}
for lab,(ls,c,cls) in runs.items():
    # 5-seed mean trajectories, so these panels use the same protocol as Table III
    # and the post-shift powers quoted in the text (5.57 / 2.92 / 2.81 mW).
    rs=[RS.run_shift(cls,F.clone(base,V=200.0,seed=sd),ts,T,sh) for sd in range(5)]
    Q=np.mean([r[0] for r in rs],axis=0); pp=np.mean([r[1] for r in rs],axis=0)
    QP[lab]=(Q,pp,ls,c)
    ax[0].plot(mavg(pp),ls=ls,color=c,lw=1.4,label=lab)
    print(f"  shift[{lab:15s}] post-shift power = {pp[ts+500:].mean():.2f} mW")
ax[0].axhline(bud,ls=':',color='k'); ax[0].axvline(ts,ls=':',color='gray')
ax[0].set_xlabel('slot $t$'); ax[0].set_ylabel('per-UAV power (mW)'); ax[0].legend(fontsize=6); ax[0].set_title('(a) Energy vs budget under shift')
for lab,(Q,pp,ls,c) in QP.items(): ax[1].plot(Q,ls=ls,color=c,lw=1.4,label=lab)
ax[1].axvline(ts,ls=':',color='gray'); ax[1].set_xlabel('slot $t$'); ax[1].set_ylabel(r'energy backlog $\sum_n Q_n$'); ax[1].set_title('(b) Feasibility: backlog bounded vs diverging')

STY=[('-','o','#1f77b4'),('--','s','#d62728'),(':','^','#2ca02c')]
key=[F.MaxWeight,F.DPP_NoValue,F.FreshSkyDPP]; nm=['Max-Weight (infeas.)','Lyap-DPP','FreshSky*']
# (c) blockage burstiness -- 5-seed mean +/- std
xs=[2,4,6,8,12,16]
for i,P in enumerate(key):
    ys,es=sweep_ms(P,[dict(mean_sojourn_N=x) for x in xs])
    ls,mk,c=STY[i]; ax[2].errorbar(xs,ys,yerr=es,fmt=ls+mk,color=c,label=nm[i],capsize=2,ms=4,lw=1.2,elinewidth=.8)
ax[2].set_xlabel('mean NLoS dwell (burstiness)'); ax[2].set_ylabel('weighted AoI'); ax[2].legend(fontsize=6); ax[2].set_title('(c) Robustness to correlated blockage')
# (d) interference -- 5-seed mean +/- std
xs=[0.1,0.2,0.3,0.4,0.5]
for i,P in enumerate(key):
    ys,es=sweep_ms(P,[dict(p_ihigh=x) for x in xs])
    ls,mk,c=STY[i]; ax[3].errorbar(xs,ys,yerr=es,fmt=ls+mk,color=c,label=nm[i],capsize=2,ms=4,lw=1.2,elinewidth=.8)
ax[3].set_xlabel('interference prob. $p_I$'); ax[3].set_ylabel('weighted AoI'); ax[3].legend(fontsize=6); ax[3].set_title('(d) Robustness to interference')
# (e) scalability -- extend to N=50, 5-seed mean +/- std
Ns=[6,9,12,16,20,32,50]
for i,P in enumerate(key):
    ys,es=sweep_ms(P,[dict(N=n,M=max(2,n//4)) for n in Ns])
    ls,mk,c=STY[i]; ax[4].errorbar(Ns,ys,yerr=es,fmt=ls+mk,color=c,label=nm[i],capsize=2,ms=4,lw=1.2,elinewidth=.8)
ax[4].set_xlabel('number of UAVs $N$ ($M{=}N/4$)'); ax[4].set_ylabel('weighted AoI'); ax[4].legend(fontsize=6); ax[4].set_title('(e) Scalability to $N{=}50$')
# (f) event-detection latency -- 5-seed mean +/- std bars
lst=[F.FreshSkyDPP,F.DPP_NoValue,F.MaxWeight,F.MaxAgeFirst,F.Opportunistic]; nm2=['FreshSky*','Lyap-DPP','Max-Weight','Max-Age','Opportun.']
resL=[F.run_multi(P,F.clone(base,V=200.0),seeds=SEEDS5) for P in lst]
dl=[r['det_latency'] for r in resL]; dle=[r['det_latency_std'] for r in resL]; ok=[r['energy_ok'] for r in resL]
bb=ax[5].bar(range(len(lst)),dl,yerr=dle,capsize=3,color=['#2ca02c' if o else '#d62728' for o in ok],edgecolor='k',lw=.5,error_kw=dict(elinewidth=.8))
for b,o in zip(bb,ok):
    if not o: b.set_hatch('////')
ax[5].set_xticks(range(len(lst))); ax[5].set_xticklabels(nm2,rotation=25,ha='right',fontsize=6.5)
ax[5].set_ylabel('event-detection latency'); ax[5].set_title('(f) Time from event to fresh delivery')
fig.tight_layout(pad=0.6); fig.savefig(os.path.join(FIG,'fig_grid_robust.png'),dpi=155,bbox_inches='tight'); plt.close(fig)
print('saved fig_grid_robust.png')
