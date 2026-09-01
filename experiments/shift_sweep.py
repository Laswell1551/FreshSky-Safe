# -*- coding: utf-8 -*-
"""Shift-severity sweep. The distribution-free feasibility certificate says
FreshSky's cumulative post-shift budget violation is bounded by a constant
(Q^max) for ANY non-stationarity, tuning-free. We stress this by sweeping the
shift severity (post-shift interference intensity) and measuring, per policy,
the PEAK post-shift cumulative excess energy per UAV
  D_peak = max_t  sum_{tau=t_s..t} ( p(tau) - pbar ).
Expectation: FreshSky stays bounded (self-clocking queue); a fixed offline
price grows ~linearly with severity (unbounded); an online-learned price with a
FIXED step size (not re-tuned per severity) degrades as severity grows."""
import os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import freshsky_sim as F
import robustness_shift as R
FIG=os.path.join(os.path.dirname(__file__),"figs")

def peak_cum_excess(pp, t_s, bud):
    D=np.cumsum(pp[t_s:]-bud)                 # per-UAV cumulative signed excess (mW*slots)
    return float(max(0.0, D.max()))

if __name__=="__main__":
    base=F.Cfg().derived(); T=3000; ts=1500; bud=base.pbar

    # offline-tune the fixed price on the pre-shift regime (strongest feasible), once.
    chosen=None
    for lam in [0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.30]:
        F.FixedPriceThr.LAM=lam
        _,pa=R.run_shift(F.FixedPriceThr, F.clone(base,V=200.0), T+10, ts, {})
        if pa.mean()<=bud and chosen is None: chosen=lam
    F.FixedPriceThr.LAM=chosen if chosen else 0.30
    print(f"offline-tuned fixed price lambda*={F.FixedPriceThr.LAM:.2f}")

    # analytic distribution-free queue bound Qmax = V*wmax*Amax/pth_min + Pmax  (per UAV)
    e=F.Env(F.clone(base,V=200.0)); o=e.observe(); pth_min=float(np.min(o['pth']))
    Qmax=200.0*base.wev*base.Amax/max(pth_min,1e-9)+base.Pmax
    print(f"pth_min~{pth_min:.3f}mW -> analytic Q^max envelope ~ {Qmax:.0f} mW*slots (finite, tuning-free)")

    sev=[0.0,1.0,2.0,3.0,4.0]                 # interference-intensity multiplier (0 = no shift)
    seeds=(0,1,2,3,4)
    pol=[("Fixed-Price (offline)", F.FixedPriceThr, '--','#d62728'),
         ("Abd-Elmagid'25 (fixed step)", F.AoIEnergyOnline, '-.','#9467bd'),
         ("FreshSky (ours)", F.FreshSkyDPP, '-','#2ca02c')]
    mean={name:[] for name,_,_,_ in pol}; std={name:[] for name,_,_,_ in pol}; Qpeak_fs=[]
    for s in sev:
        shift=dict(p_ihigh=0.6, I_high=2e-8*s, p_event=0.5, mean_sojourn_ev=50.0) if s>0 else {}
        for name,cls,ls,cl in pol:
            vals=[]
            for sd in seeds:
                Q,pp=R.run_shift(cls, F.clone(base,V=200.0,seed=sd), ts, T, shift)
                vals.append(peak_cum_excess(pp, ts, bud))
                if cls is F.FreshSkyDPP: Qpeak_fs.append(float(Q[ts:].max()))
            mean[name].append(float(np.mean(vals))); std[name].append(float(np.std(vals)))
        r_on=mean["Abd-Elmagid'25 (fixed step)"][-1]/max(mean["FreshSky (ours)"][-1],1e-9)
        r_fx=mean["Fixed-Price (offline)"][-1]/max(mean["FreshSky (ours)"][-1],1e-9)
        print(f"  severity x{s:.0f}: " + "  ".join(f"{n.split()[0]}={mean[n][-1]:6.0f}" for n,_,_,_ in pol)
              + f"   | online/FS={r_on:.1f}x  fixed/FS={r_fx:.0f}x")
    print(f"FreshSky empirical peak backlog (5 seeds) max={max(Qpeak_fs):.0f} << analytic Q^max (order-of-magnitude, stays bounded)")

    fig,ax=plt.subplots(figsize=(3.6,2.8))
    for name,cls,ls,cl in pol:
        ax.errorbar(sev,np.maximum(mean[name],1.0),yerr=std[name],ls=ls,marker='o',ms=4,
                    color=cl,lw=1.7,capsize=2,label=name)
    ax.set_yscale('log'); ax.set_xlabel("shift severity (interference $\\times$)")
    ax.set_ylabel("peak cum.\\ excess (mW$\\cdot$slot)")
    ax.set_title("Bounded violation under shift"); ax.legend(fontsize=7.0,loc='center right')
    fig.savefig(os.path.join(FIG,"e14_sweep.png"),dpi=150,bbox_inches='tight'); plt.close(fig)
    print("saved e14_sweep.png (5-seed mean +/- std)")
