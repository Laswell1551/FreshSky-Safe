# -*- coding: utf-8 -*-
"""Robustness under distribution shift. At t_shift a channel+event shift hits
(heavier interference AND more critical regions). We compare three energy-aware
schedulers, all honestly parameterized:
  (1) Fixed-Price Thr.        -- energy price tuned OFFLINE on the pre-shift
                                 regime (best feasible price there); it cannot
                                 react and VIOLATES the budget after the shift.
  (2) AoI-Energy Online [Abd-Elmagid'25] -- learns the price online (dual
                                 ascent); it RE-ADAPTS and recovers feasibility
                                 after a transient, at the cost of a tuned step.
  (3) FreshSky (ours)         -- virtual-queue price adapts with a PROVABLE
                                 feasibility guarantee (Thm 1) and no step to
                                 tune.
No price is hand-picked to fail: the fixed price is the most aggressive one that
is feasible before the shift. The point is that a fixed price is stale under
non-stationarity, while adaptive prices are not -- and FreshSky is adaptive AND
guaranteed AND tuning-free."""
import os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import freshsky_sim as F
FIG=os.path.join(os.path.dirname(__file__),"figs")

def run_shift(PolicyCls, cfg, t_shift, T, shift):
    cfg=F.clone(cfg,T=T); env=F.Env(cfg); pol=PolicyCls(cfg)
    Q=[]; ppUAV=[]
    for t in range(T):
        if t==t_shift:
            for k,v in shift.items(): setattr(env.cfg,k,v)
        obs=env.observe(); env._last=obs
        sched,power=pol.decide(env,obs); s,p=env.step(sched,power)
        Q.append(float(env.Q.sum())); ppUAV.append(float(p.sum())/cfg.N)
    return np.array(Q), np.array(ppUAV)

def moving_avg(x,w=100):
    c=np.cumsum(np.insert(x,0,0)); return (c[w:]-c[:-w])/w

if __name__=="__main__":
    base=F.Cfg().derived(); T=3000; ts=1500; bud=base.pbar
    shift=dict(p_ihigh=0.6, I_high=4e-8, p_event=0.5, mean_sojourn_ev=50.0)

    # --- offline-tune the fixed price on the PRE-shift regime: pick the most
    #     aggressive (lowest-AoI) lambda that is still feasible before the shift.
    print("== offline-tuning the fixed price on the pre-shift regime ==")
    chosen=None
    for lam in [0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40]:
        F.FixedPriceThr.LAM=lam
        _,pa=run_shift(F.FixedPriceThr, F.clone(base,V=200.0), T+10, ts, {})  # no shift
        pre=pa.mean(); feas=pre<=bud
        print(f"  lambda={lam:.2f}  pre-shift pow={pre:.2f} mW  feasible={feas}")
        if feas and chosen is None: chosen=lam        # smallest feasible lambda
    if chosen is None: chosen=0.40
    F.FixedPriceThr.LAM=chosen
    print(f"  -> offline-tuned fixed price lambda*={chosen:.2f}")

    print("== running the shift (t_shift=%d) ==" % ts)
    pols=[("Fixed-Price (offline-tuned)", F.FixedPriceThr, '--', '#d62728'),
          ("Abd-Elmagid'25 (online)",     F.AoIEnergyOnline, '-.', '#9467bd'),
          ("FreshSky (ours)",             F.FreshSkyDPP,    '-',  '#2ca02c')]
    res={}
    for label,cls,ls,cl in pols:
        Q,pp=run_shift(cls, F.clone(base,V=200.0), ts, T, shift)
        pre=pp[:ts].mean(); trans=pp[ts:ts+500].mean(); steady=pp[ts+500:].mean()
        res[label]=(Q,pp,ls,cl)
        print(f"  {label:30s} pre={pre:5.2f}  transient={trans:5.2f}  "
              f"steady-post={steady:5.2f} mW  feasible_post={steady<=bud*1.02}")
    print(f"  budget per-UAV = {bud:.2f} mW")

    fig,ax=plt.subplots(1,2,figsize=(8.4,3.2))
    for label,(Q,pp,ls,cl) in res.items():
        mf=moving_avg(pp); ax[0].plot(np.arange(len(mf)),mf,ls=ls,color=cl,lw=1.7,label=label)
    ax[0].axhline(bud,ls=':',color='k',label=f"budget $\\bar p$={bud:.0f} mW")
    ax[0].axvline(ts,ls=':',color='gray'); ax[0].text(ts+40,ax[0].get_ylim()[1]*0.55,"shift",fontsize=8,color='gray')
    ax[0].set_xlabel("slot $t$"); ax[0].set_ylabel("per-UAV avg power (mW)")
    ax[0].legend(fontsize=7.5,loc='upper left'); ax[0].set_title("(a) Energy vs budget")
    for label,(Q,pp,ls,cl) in res.items():
        ax[1].plot(Q,ls=ls,color=cl,lw=1.7,label=label)
    ax[1].axvline(ts,ls=':',color='gray'); ax[1].text(ts+40,ax[1].get_ylim()[1]*0.55,"shift",fontsize=8,color='gray')
    ax[1].set_xlabel("slot $t$"); ax[1].set_ylabel(r"energy backlog $\sum_n Q_n(t)$")
    ax[1].legend(fontsize=7.5,loc='upper left'); ax[1].set_title("(b) Feasibility under shift")
    fig.tight_layout(); fig.savefig(os.path.join(FIG,"e8_shift.png"),dpi=150,bbox_inches='tight'); plt.close(fig)
    print("saved e8_shift.png")
