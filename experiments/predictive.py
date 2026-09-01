# -*- coding: utf-8 -*-
"""PROTOTYPE: trajectory-predictive scheduling (FreshSky-T) on REAL OSM geometry.
Idea: a UAV flies a KNOWN mission trajectory, and with a city map blockage is
PREDICTABLE. FreshSky-T folds the predicted imminent-outage length G_n (slots
the link will be dark if skipped now) into the index -- serving a link BEFORE it
goes behind a building. Baselines track the channel REACTIVELY (belief) or not
at all (price). Same air-ground channel + delivery code as the paper (imported).
Comparison at a glance: does prediction beat belief-Whittle / Abd-Elmagid on real
predictable geometry?"""
import sys, numpy as np
import urban_blockage as U

MODE = sys.argv[1] if len(sys.argv)>1 else 'osm'     # 'osm' | 'proc'
NREQ = int(sys.argv[2]) if len(sys.argv)>2 else 12
if MODE=='proc':
    # procedural city, tunable N (LoS ~0.74 -> more available links -> contention for M bands)
    los,dist=U.gen_traces(N=NREQ, T=2000, seed=1)
else:
    import osm_urban as O
    data,_=O.fetch_or_load(); polys,heights,src=O.parse_buildings(data)
    H,cell,sx,sy,proj=O.build_grid(polys,heights); gs,_=O.place_gs(H,cell)
    los,dist=O.gen_traces_osm(H,cell,sx,sy,gs)        # [T,N] true LoS + range
T,N=los.shape
FC,ETAL,PMAX,SIGMA2,GAMMA,PL,PN=U.FC,U.ETAL,U.PMAX,U.SIGMA2,U.GAMMA,U.PHI_LOS,U.PHI_NLOS

def imminent_outage(los, Hh=20):
    """G[t,n] = length of the NLoS run starting at t+1 (capped at Hh):
    how many slots UAV n stays blocked if not served now."""
    T,N=los.shape; run=np.zeros((T,N),int)
    for t in range(T-2,-1,-1): run[t]=np.where(~los[t],1+run[t+1],0)
    G=np.zeros((T,N),int); G[:-1]=np.minimum(run[1:],Hh); return G
G=imminent_outage(los)

def run(sched, V=6.0, pbar=1.6, wev=5.0, p_event=0.08, ev_off=1/25,
        Amax=120, warmup=300, M=3, seed=0, kappa=1.0):
    rng=np.random.default_rng(9000+seed)
    fspl=20*np.log10(4*np.pi*FC*dist/3e8)
    pth=np.minimum(GAMMA*SIGMA2/10**(-(fspl+ETAL)/10),PMAX)
    piL=np.clip(los.mean(0),0.05,0.95); pNL=1/8.0; pLN=np.clip(pNL*(1-piL)/piL,0,1)
    th=piL.copy(); lam=np.full(N,0.25); A=np.ones(N); Q=np.zeros(N); ev=(rng.random(N)<p_event)
    wA=0.0;cnt=0;esum=0.0
    for t in range(T):
        w=np.where(ev,wev,1.0); ld=los[t].astype(float)
        if   sched=='memoryless': idx=V*w*A - Q*pth[t]
        elif sched=='belief':     idx=V*w*A*th*PL - Q*pth[t]
        elif sched=='abdel':      idx=w*A - lam*pth[t]
        elif sched=='genie':      idx=V*w*A*ld*PL - Q*pth[t]                 # knows current LoS
        elif sched=='predict':    idx=V*w*(A+kappa*G[t])*ld*PL - Q*pth[t]    # map + lookahead
        order=np.argsort(-idx); S=[int(n) for n in order[:M] if idx[n]>0]
        e=np.zeros(N);ack=np.zeros(N,bool);att=np.zeros(N,bool)
        for n in S:
            e[n]=pth[t,n];att[n]=True; ack[n]=rng.random()<(PL if los[t,n] else PN)
        A=np.where(ack,1.0,np.minimum(A+1,Amax)); Q=np.maximum(Q-pbar,0.0)+e
        post=th.copy()
        for n in range(N):
            if att[n]:
                if ack[n]: post[n]=th[n]*PL/(th[n]*PL+(1-th[n])*PN+1e-12)
                else:      post[n]=th[n]*(1-PL)/(th[n]*(1-PL)+(1-th[n])*(1-PN)+1e-12)
        th=post*(1-pLN)+(1-post)*pNL
        lam=np.maximum(lam+3e-3*(e-pbar),0.0)
        u=rng.random(N); ev=np.where(ev,u>=ev_off,u<p_event)
        if t>=warmup: wA+=float(np.sum(w*A));cnt+=1;esum+=float(np.sum(e))
    return wA/cnt, esum/cnt/N

if __name__=="__main__":
    print(f"REAL OSM Manhattan: T={T} N={N}  LoS={los.mean():.3f}  "
          f"autocorr@1={U.autocorr(los[:,0])[0]:.2f}  mean imminent-outage={G[G>0].mean():.1f} slots")
    res={}
    for name in ['memoryless','belief','abdel','genie','predict']:
        wa,en=np.mean([run(name,seed=s) for s in range(4)],axis=0); res[name]=(wa,en)
        print(f"  {name:12s} wAoI={wa:8.1f}  pow/UAV={en:.3f} mW")
    b=res['belief']; p=res['predict']; a=res['abdel']; g=res['genie']
    print(f"\n  FreshSky-T vs Belief-Whittle: AoI {100*(b[0]-p[0])/b[0]:+.1f}%  power {100*(b[1]-p[1])/b[1]:+.1f}%")
    print(f"  FreshSky-T vs Abd-Elmagid:    AoI {100*(a[0]-p[0])/a[0]:+.1f}%  power {100*(a[1]-p[1])/a[1]:+.1f}%")
    print(f"  FreshSky-T vs Genie(now-only):AoI {100*(g[0]-p[0])/g[0]:+.1f}%  power {100*(g[1]-p[1])/g[1]:+.1f}%")
