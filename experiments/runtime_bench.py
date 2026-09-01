# -*- coding: utf-8 -*-
"""Real-hardware runtime measurement of FreshSky's per-slot scheduling cost.
Not a UAV testbed -- a genuine implementation measurement on this machine,
validating the O(N log N) / real-time claim. Times the decide() call vs N."""
import os, time, platform, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import freshsky_sim as F
FIG=os.path.join(os.path.dirname(__file__),"figs")

def synth_obs(N, rng):
    return dict(A=rng.integers(1,50,N).astype(float),
                Q=rng.uniform(0,40,N),
                w=rng.choice([1.0,10.0],N),
                pth=rng.uniform(0.5,120,N),
                deliverable=(rng.random(N)<0.63),
                g=rng.uniform(1e-11,1e-8,N), I=np.zeros(N),
                beta=rng.integers(0,2,N), theta=rng.uniform(15,60,N))

def bench(PolicyCls, N, reps, rng):
    cfg=F.clone(F.Cfg().derived(), N=N, M=max(2,N//4), V=200.0)
    pol=PolicyCls(cfg); obs=synth_obs(N,rng)
    # warmup
    for _ in range(50): pol.decide(None,obs)
    t=[]
    for _ in range(reps):
        obs=synth_obs(N,rng)
        t0=time.perf_counter(); pol.decide(None,obs); t.append(time.perf_counter()-t0)
    return np.median(t)*1e6   # microseconds

if __name__=="__main__":
    rng=np.random.default_rng(0)
    print("Machine:", platform.processor() or platform.machine(), "|", platform.system(), platform.release())
    Ns=[12,25,50,100,200,500,1000,2000]
    us=[bench(F.FreshSkyDPP, N, 3000, rng) for N in Ns]
    for N,u in zip(Ns,us): print(f"  N={N:5d}  FreshSky decide = {u:8.2f} us  ({u/1e3:.3f} ms)")
    # figure: runtime vs N with N log N reference
    fig,ax=plt.subplots(figsize=(5.2,3.2))
    ax.plot(Ns,us,'-o',color='#2ca02c',lw=1.8,label="FreshSky (measured)")
    ref=np.array(Ns,float); ref=ref*np.log2(ref); ref=ref/ref[0]*us[0]
    ax.plot(Ns,ref,'--',color='0.5',label=r"$O(N\log N)$ reference")
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel("number of UAVs $N$"); ax.set_ylabel(r"per-slot scheduling time ($\mu$s)")
    ax.legend(fontsize=9); ax.set_title("Measured scheduling cost (commodity CPU)")
    fig.savefig(os.path.join(FIG,"e9_runtime.png"),dpi=150,bbox_inches='tight'); plt.close(fig)
    print("saved e9_runtime.png")
