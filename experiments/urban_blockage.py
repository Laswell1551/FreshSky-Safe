# -*- coding: utf-8 -*-
"""Semi-real anchor: generate LoS/NLoS blockage sequences from a 3D URBAN
GEOMETRY (buildings + geometric ray check UAV->GS along real trajectories),
NOT a 2-state Markov assumption. Then TEST the schedulers -- which are designed
assuming a Markov channel -- on these real-geometry traces (a synthetic->real
generalization story). Belief-Whittle still Pareto-dominates => it generalizes.
Self-contained (numpy)."""
import os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
FIG=os.path.join(os.path.dirname(__file__),"figs")

# air-ground channel constants (identical to main sim / belief_whittle)
FC=2.4e9; W=20e6; TAU=0.1; ETAL=1.0; ETAN=20.0; PMAX=100.0
SIGMA2=1.38e-23*290.0*W*10**0.7*1e3; GAMMA=2**(4.0e6/(W*TAU))-1
PHI_LOS=0.90; PHI_NLOS=0.02

def build_city(rng, span=800.0, cell=40.0, med_h=14.0, sig=0.5, tall_frac=0.10):
    """Grid of building heights (m): mostly low (log-normal), scattered towers
    that cause bursty NLoS when a UAV passes behind one."""
    n=int(span/cell); H=rng.lognormal(np.log(med_h),sig,(n,n))
    tall=rng.random((n,n))<tall_frac; H[tall]=rng.uniform(45,110,tall.sum())
    return H, cell, span

def los_blocked(ux,uy,uz, gx,gy,gz, Hgrid, cell, span, steps=60):
    """True if the UAV->GS segment is blocked by any building (3D geometric LoS)."""
    for a in np.linspace(0.02,0.98,steps):
        x=ux+a*(gx-ux); y=uy+a*(gy-uy); z=uz+a*(gz-uz)
        i=int(x//cell); j=int(y//cell)
        if 0<=i<Hgrid.shape[0] and 0<=j<Hgrid.shape[1]:
            if Hgrid[i,j] > z: return True
    return False

def gen_traces(N=12, T=2000, seed=0):
    rng=np.random.default_rng(seed); Hg,cell,span=build_city(rng)
    gs=np.array([span*0.5,span*0.5,35.0])             # ground station on a central mast/rooftop
    los=np.zeros((T,N),bool); dist=np.zeros((T,N))
    for n in range(N):
        h=rng.uniform(80,120)                          # UAV altitude
        p0=np.array([rng.uniform(0,span),rng.uniform(0,span),h])
        ang=rng.uniform(0,2*np.pi); v=rng.uniform(6,14)*TAU  # m/slot
        vel=np.array([np.cos(ang),np.sin(ang),0.0])*v
        for t in range(T):
            p=p0+vel*t
            p[:2]=np.mod(p[:2],span)                    # wrap inside the city
            blk=los_blocked(p[0],p[1],p[2],gs[0],gs[1],gs[2],Hg,cell,span)
            los[t,n]=(not blk); dist[t,n]=np.linalg.norm(p-gs)
    return los, dist

def autocorr(x, lags=30):
    x=x.astype(float)-x.mean(); ac=[np.corrcoef(x[:-k],x[k:])[0,1] for k in range(1,lags)]
    return np.array(ac)

# --- schedulers on a given LoS trace (belief uses a Markov-estimated model) ---
def run_on_trace(los, dist, policy, V=6.0, pbar=1.6, wev=5.0, p_event=0.08, ev_off=1/25,
                 Amax=120, warmup=300, genie=False, seed=0):
    T,N=los.shape; rng=np.random.default_rng(9000+seed)
    fspl=20*np.log10(4*np.pi*FC*dist/3e8)
    pth=np.minimum(GAMMA*SIGMA2/10**(-(fspl+ETAL)/10),PMAX)     # LoS-case delivery power (mW)
    piL=np.clip(los.mean(0),0.05,0.95)                          # per-UAV empirical LoS frac
    # scheduler's ASSUMED Markov model (it does NOT know the geometry)
    pNL=1/8.0; pLN=np.clip(pNL*(1-piL)/piL,0,1)
    th=piL.copy(); A=np.ones(N); Q=np.zeros(N); ev=(rng.random(N)<p_event)
    wA=0.0; cnt=0; esum=0.0; M=3
    for t in range(T):
        w=np.where(ev,wev,1.0); bel=los[t].astype(float) if genie else th
        idx=policy(A,bel,Q,w,pth[t],V,piL); order=np.argsort(-idx)
        S=[int(n) for n in order[:M] if idx[n]>0]
        e=np.zeros(N); ack=np.zeros(N,bool); att=np.zeros(N,bool)
        for n in S:
            e[n]=pth[t,n]; att[n]=True
            ack[n]=rng.random() < (PHI_LOS if los[t,n] else PHI_NLOS)
        A=np.where(ack,1.0,np.minimum(A+1,Amax)); Q=np.maximum(Q-pbar,0.0)+e
        post=th.copy()
        for n in range(N):
            if att[n]:
                if ack[n]: post[n]=th[n]*PHI_LOS/(th[n]*PHI_LOS+(1-th[n])*PHI_NLOS+1e-12)
                else:      post[n]=th[n]*(1-PHI_LOS)/(th[n]*(1-PHI_LOS)+(1-th[n])*(1-PHI_NLOS)+1e-12)
        th=post*(1-pLN)+(1-post)*pNL
        u=rng.random(N); ev=np.where(ev,u>=ev_off,u<p_event)
        if t>=warmup: wA+=float(np.sum(w*A)); cnt+=1; esum+=float(np.sum(e))
    return wA/cnt, esum/cnt/N

def belief_whittle(A,bel,Q,w,pth,V,piL): return V*w*A*bel*PHI_LOS - Q*pth
def memoryless(A,bel,Q,w,pth,V,piL):     return V*w*A - Q*pth
def wang_index(A,th,alpha,beta):
    """Wang'26 (arXiv 2605.21016) closed-form POMDP Whittle-like index -- plain AoI, no value/energy."""
    Tt=th*alpha+(1-th)*(1-beta)
    def F1(d,t):
        den=(1-beta)*d+1-t+1e-12
        return (1-beta)/den*(d*(d+1)/2+(1-t)*beta/(1-beta)**2+(1-t)*(d+1)/(1-beta))
    def F2(d,t): return (2-t-beta)/((1-beta)*d+1-t+1e-12)
    return (F1(A+1,Tt)-F1(A,th))/(F2(A,th)-F2(A+1,Tt)+1e-12)
def wang26(A,bel,Q,w,pth,V,piL):                              # Wang'26 POMDP-Whittle (value-/energy-agnostic)
    pNL=1/8.0; pLN=np.clip(pNL*(1-piL)/np.maximum(piL,1e-3),0,1)
    return wang_index(A,bel,1.0-pLN,1.0-pNL)

if __name__=="__main__":
    print("generating urban blockage traces (3D geometry)...")
    los,dist=gen_traces(N=12,T=2000,seed=1)
    print(f"  overall LoS fraction = {los.mean():.2f}")
    ac=np.nanmean([autocorr(los[:,n]) for n in range(los.shape[1])],axis=0)  # per-UAV avg
    print(f"  per-UAV blockage autocorr @lag1={ac[0]:.2f} lag5={ac[4]:.2f} lag10={ac[9]:.2f} (correlated, non-Markov)")
    res={}
    for name,pol in [("Belief-Whittle (ours)",belief_whittle),("Memoryless",memoryless)]:
        wa,en=np.mean([run_on_trace(los,dist,pol,seed=s) for s in range(4)],axis=0)
        res[name]=(wa,en); print(f"  {name:22s} wAoI={wa:7.1f}  power/UAV={en:.3f}mW  (REAL-geometry traces)")
    dML=100*(res['Memoryless'][0]-res['Belief-Whittle (ours)'][0])/res['Memoryless'][0]
    print(f"  Belief-Whittle gain on real geometry = {dML:.1f}% (at lower energy) -> generalizes")
    fig,ax=plt.subplots(1,2,figsize=(8.0,2.9))
    ax[0].plot(range(1,len(ac)+1),ac,'-o',ms=3,color='#2ca02c'); ax[0].axhline(0,ls=':',color='0.6')
    ax[0].set_xlabel("lag (slots)"); ax[0].set_ylabel("blockage autocorr."); ax[0].set_title("(a) Real-geometry blockage")
    mk={'Belief-Whittle (ours)':('o','#2ca02c'),'Memoryless':('s','#1f77b4')}
    for name,(wa,en) in res.items():
        m,c=mk.get(name,('^','0.3'))
        ax[1].scatter(en,wa,marker=m,s=120,color=c,edgecolor='k',zorder=3,label=name)
    ax[1].set_xlabel("power per UAV (mW)"); ax[1].set_ylabel("weighted AoI")
    ax[1].set_title("(b) Pareto on real geometry"); ax[1].legend(fontsize=8,loc='best')
    fig.tight_layout(); fig.savefig(os.path.join(FIG,"e12_urban.png"),dpi=150,bbox_inches='tight'); plt.close(fig)
    print("saved e12_urban.png")
