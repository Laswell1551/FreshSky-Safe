# -*- coding: utf-8 -*-
"""FreshSky-A: JOINT altitude-and-scheduling co-design over blockage-prone
air-ground channels. Prior AoI schedulers fix the UAV altitude and schedule
AROUND blockage; FreshSky-A makes altitude a decision variable and climbs stale,
high-value UAVs into line-of-sight -- attacking the outages at the source.

Reproducible: the (slow) LoS/distance ray-marching over the real 3D geometry is
cached to data/altcube_<mode>.npz, so re-runs are fast and offline. Channel +
belief + delivery reuse the paper's code (urban_blockage). Emits e15_altitude.png
and the headline numbers.  Run: python joint_altitude.py [osm|proc]"""
import os, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import urban_blockage as U
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data"); FIG=os.path.join(HERE,"figs")
os.makedirs(DATA,exist_ok=True)
MODE=sys.argv[1] if len(sys.argv)>1 else 'osm'
N=12; T=1200; ALT=[80,120,160,200,260,320]; PL,PN=U.PHI_LOS,U.PHI_NLOS; ALT_arr=np.array(ALT,float)

def geometry():
    if MODE=='osm':
        import osm_urban as O
        data,_=O.fetch_or_load(); polys,heights,_=O.parse_buildings(data)
        Hg,cell,sx,sy,_=O.build_grid(polys,heights); gs,_=O.place_gs(Hg,cell)
        return Hg,cell,sx,sy,gs,heights.max()
    rng=np.random.default_rng(1); Hg,cell,span=U.build_city(rng)
    return Hg,cell,span,span,np.array([span*0.5,span*0.5,35.0]),Hg.max()

def build_cube():
    """LoS[T,N,K] + pth[T,N,K] (LoS-case delivery power) over the real geometry, cached."""
    cache=os.path.join(DATA,f"altcube_{MODE}.npz")
    if os.path.exists(cache):
        z=np.load(cache); print(f"[cache] {os.path.relpath(cache,HERE)}"); return z['LoS'],z['pth'],float(z['tallest'])
    Hg,cell,sx,sy,gs,tallest=geometry()
    rng=np.random.default_rng(1); xy=np.zeros((T,N,2))
    for n in range(N):
        p0=np.array([rng.uniform(0,sx),rng.uniform(0,sy)]); ang=rng.uniform(0,2*np.pi); v=rng.uniform(6,14)*U.TAU
        vel=np.array([np.cos(ang),np.sin(ang)])*v
        for t in range(T): xy[t,n]=(p0+vel*t)%[sx,sy]
    LoS=np.zeros((T,N,len(ALT)),bool); D=np.zeros((T,N,len(ALT)))
    for t in range(T):
        for n in range(N):
            for k,zz in enumerate(ALT):
                seg=np.sqrt((xy[t,n,0]-gs[0])**2+(xy[t,n,1]-gs[1])**2+(zz-gs[2])**2)
                steps=int(np.clip(seg/(cell*0.7),60,400))
                LoS[t,n,k]=not U.los_blocked(xy[t,n,0],xy[t,n,1],zz,gs[0],gs[1],gs[2],Hg,cell,sx,steps=steps)
                D[t,n,k]=seg
    fspl=20*np.log10(4*np.pi*U.FC*D/3e8); pth=np.minimum(U.GAMMA*U.SIGMA2/10**(-(fspl+U.ETAL)/10),U.PMAX)
    np.savez_compressed(cache,LoS=LoS,pth=pth,tallest=tallest)
    print(f"[built+cached] {os.path.relpath(cache,HERE)}"); return LoS,pth,tallest

LoS,pth,tallest=build_cube()

def run(mode, base='belief', V=6.0, pbar=1.6, wev=5.0, p_event=0.08, ev_off=1/25, Amax=120,
        warmup=300, M=3, seed=0, fk=1, dz_mps=12.0, rho=0.02):
    """mode 'fixed': baseline scheduler `base` cruising at altitude fk.
       mode 'joint': FreshSky-A -- per-UAV altitude (rate-limited dz_mps, propulsion
       penalty rho) + value*age*LoS scheduling. Returns (wAoI, tx pow/UAV, climb m/s)."""
    rng=np.random.default_rng(9000+seed); dz=dz_mps*U.TAU
    piL=np.clip(LoS[:,:,fk].mean(0),0.05,0.95); pNL=1/8.0; pLN=np.clip(pNL*(1-piL)/piL,0,1)
    th=piL.copy(); lam=np.full(N,0.25); A=np.ones(N); Q=np.zeros(N); ev=(rng.random(N)<p_event)
    z=np.full(N,float(ALT[fk])); wA=0.0;cnt=0;esum=0.0;climb=0.0
    for t in range(T):
        w=np.where(ev,wev,1.0)
        if mode=='joint':
            idxk=V*(w*A)[:,None]*LoS[t]*PL - Q[:,None]*pth[t] - rho*np.abs(ALT_arr[None,:]-z[:,None])
            kdes=np.argmax(idxk,1); d=np.clip(ALT_arr[kdes]-z,-dz,dz); z=z+d; climb+=float(np.abs(d).sum())
            ku=np.clip(np.round((z-ALT[0])/40.0).astype(int),0,len(ALT)-1)
            idx=V*w*A*LoS[t,np.arange(N),ku]*PL - Q*pth[t,np.arange(N),ku]
        else:
            ku=np.full(N,fk)
            if   base=='belief':     idx=V*w*A*th*PL - Q*pth[t,:,fk]
            elif base=='memoryless': idx=V*w*A - Q*pth[t,:,fk]
            elif base=='abdel':      idx=w*A - lam*pth[t,:,fk]
        order=np.argsort(-idx); S=[int(n) for n in order[:M] if idx[n]>0]
        e=np.zeros(N);ack=np.zeros(N,bool);att=np.zeros(N,bool)
        for n in S:
            k=ku[n]; e[n]=pth[t,n,k]; att[n]=True; ack[n]=rng.random()<(PL if LoS[t,n,k] else PN)
        A=np.where(ack,1.0,np.minimum(A+1,Amax)); Q=np.maximum(Q-pbar,0.0)+e
        post=th.copy()
        for n in range(N):
            if att[n]:
                if ack[n]: post[n]=th[n]*PL/(th[n]*PL+(1-th[n])*PN+1e-12)
                else:      post[n]=th[n]*(1-PL)/(th[n]*(1-PL)+(1-th[n])*(1-PN)+1e-12)
        th=post*(1-pLN)+(1-post)*pNL; lam=np.maximum(lam+3e-3*(e-pbar),0.0)
        u=rng.random(N); ev=np.where(ev,u>=ev_off,u<p_event)
        if t>=warmup: wA+=float(np.sum(w*A));cnt+=1;esum+=float(np.sum(e))
    return wA/cnt, esum/cnt/N, climb/N/T*U.TAU**-1   # m/s avg climb

def avg(*a,**k):
    r=np.array([run(*a,seed=s,**k) for s in range(5)]); return r.mean(0), r.std(0)

if __name__=="__main__":
    los0=LoS[:,:,1].mean()
    print(f"[{MODE}] tallest={tallest:.0f}m  LoS@120m={los0:.3f}  LoS@320m={LoS[:,:,-1].mean():.3f}")
    print("== fixed cruising-altitude frontier (belief-Whittle) ==")
    front=[]
    for fk,zz in enumerate(ALT):
        (wa,en,_),(sa,se,_)=avg('fixed','belief',fk=fk)
        front.append((zz,wa,en,sa)); print(f"   z={zz:3d}m  wAoI={wa:8.1f}+/-{sa:5.1f}  pow={en:.3f}mW")
    print("== baselines at their best-AoI fixed altitude (z=320) ==")
    base_pts={}
    for b,lab in [('belief','Belief-Whittle'),('memoryless','Memoryless'),('abdel',"Abd-Elmagid'25")]:
        (wa,en,_),_=avg('fixed',b,fk=len(ALT)-1); base_pts[lab]=(en,wa); print(f"   {lab:16s} wAoI={wa:8.1f}  pow={en:.3f}mW @320m")
    print("== FreshSky-A (joint altitude+scheduling, 12 m/s cap + propulsion penalty) ==")
    (wa,en,cl),(sa,se,sc)=avg('joint',dz_mps=12.0,rho=0.02)
    print(f"   FreshSky-A  wAoI={wa:8.1f}+/-{sa:.1f}  pow={en:.3f}mW  climb={cl:.1f} m/s avg")
    bw120=front[1][1]  # belief @120m
    print(f"\n   FreshSky-A vs Belief-Whittle@120m: AoI {100*(bw120-wa)/bw120:+.1f}%  (Pareto: pow {en:.3f} vs {front[1][2]:.3f})")

    # ---- figure: AoI-energy frontier + FreshSky-A dominating ----
    fig,ax=plt.subplots(figsize=(4.0,3.0))
    zz=[f[0] for f in front]; wal=[f[1] for f in front]; enl=[f[2] for f in front]; sal=[f[3] for f in front]
    ax.errorbar(enl,wal,yerr=sal,ls='-',marker='o',ms=4,color='#1f77b4',lw=1.6,capsize=2,label='Belief-Whittle, fixed alt.')
    for i,z in enumerate(zz): ax.annotate(f"{z}m",(enl[i],wal[i]),fontsize=6,xytext=(3,3),textcoords='offset points')
    mk={'Memoryless':('s','#8c564b'),"Abd-Elmagid'25":('^','#9467bd')}
    for lab,(en_,wa_) in base_pts.items():
        if lab in mk: ax.scatter(en_,wa_,marker=mk[lab][0],s=55,color=mk[lab][1],edgecolor='k',zorder=3,label=lab+' @best alt.')
    ax.scatter(en,wa,marker='*',s=240,color='#2ca02c',edgecolor='k',zorder=5,label='FreshSky-A (joint)')
    ax.annotate(f"$-${100*(bw120-wa)/bw120:.0f}% AoI",(en,wa),fontsize=8,color='#2ca02c',
                xytext=(8,-2),textcoords='offset points',fontweight='bold')
    ax.set_xlabel("transmit power per UAV (mW)"); ax.set_ylabel("weighted AoI")
    ax.set_title(f"Joint altitude control ({'Manhattan' if MODE=='osm' else 'city'})"); ax.legend(fontsize=6.4,loc='upper right')
    fig.tight_layout(); fig.savefig(os.path.join(FIG,f"e15_altitude_{MODE}.png"),dpi=150,bbox_inches='tight'); plt.close(fig)
    print(f"saved figs/e15_altitude_{MODE}.png")
