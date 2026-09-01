# -*- coding: utf-8 -*-
"""Convergence figure (running time-avg weighted AoI vs slots) + iso-energy check.
Matches the accepted-twin convention (metric-vs-time as the lead eval figure)."""
import os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import freshsky_sim as F
FIG=os.path.join(os.path.dirname(__file__),"figs")

def running_wAoI(PolicyCls, cfg, T=3000, warmup=200, seeds=(0,1,2,3,4),
                 return_stats=False):
    curves=[]
    for s in seeds:
        c=F.clone(cfg,T=T,seed=s); env=F.Env(c); pol=PolicyCls(c); inst=[]
        for t in range(T):
            obs=env.observe(); env._last=obs
            sched,power=pol.decide(env,obs); env.step(sched,power)
            inst.append(float(np.sum(obs['w']*env.A)))
        inst=np.array(inst)[warmup:]                 # drop transient (as in Table)
        curves.append(np.cumsum(inst)/np.arange(1,len(inst)+1))
    curves=np.asarray(curves)
    x=np.arange(warmup+1,T+1)
    mean=np.mean(curves,axis=0)
    if return_stats:
        sd=np.std(curves,axis=0,ddof=1) if len(curves)>1 else np.zeros_like(mean)
        return x,mean,sd
    return x,mean  # default remains backward compatible

if __name__=="__main__":
    base=F.Cfg().derived()
    series=[("FreshSky (ours)", F.FreshSkyDPP, '-', 'o', '#2ca02c', True),
            ("Abd-Elmagid'25 (online)", F.AoIEnergyOnline,'--','s', '#1f77b4', True),
            ("Lyap-DPP",        F.DPP_NoValue, ':','^', '#9467bd', True),
            ("Max-Weight (infeas.)", F.MaxWeight,'-.','D','#d62728', False)]
    fig,ax=plt.subplots(figsize=(5.6,3.4))
    for name,P,ls,mk,cl,feas in series:
        x,y=running_wAoI(P, F.clone(base,V=200.0))
        ax.plot(x,y,ls=ls,color=cl,lw=1.6,label=name)
        ax.plot(x[::400],y[::400],mk,color=cl,ms=5,ls='none')
    ax.set_xlabel("slot $t$"); ax.set_ylabel("running time-avg weighted AoI")
    ax.legend(fontsize=8,ncol=2); ax.set_title("Convergence of the objective")
    fig.savefig(os.path.join(FIG,"e0_convergence.png"),dpi=150,bbox_inches='tight'); plt.close(fig)
    print("saved e0_convergence.png")

    # iso-energy check: find LAM giving the fixed price ~ FreshSky's 2.42 mW
    print("\n== iso-energy: offline-tuned fixed price at FreshSky's power ==")
    fs=F.run_multi(F.FreshSkyDPP, F.clone(base,V=200.0), seeds=(0,1,2))
    print(f"  FreshSky V=200: wAoI={fs['wAoI']:.1f}  pow={fs['max_avg_power']:.2f}mW")
    for lam in [0.08,0.10,0.12,0.15]:
        F.FixedPriceThr.LAM=lam
        r=F.run_multi(F.FixedPriceThr, F.clone(base,V=200.0), seeds=(0,1,2))
        print(f"  Fixed-price LAM={lam:.2f}: wAoI={r['wAoI']:.1f}  pow={r['max_avg_power']:.2f}mW  feasible={r['energy_ok']}")
