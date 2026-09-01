# -*- coding: utf-8 -*-
"""FreshSky-A make-or-break: joint altitude+scheduling under a SENSING
constraint. UAVs sense the ground -> must fly low (good resolution); but low =
blocked (stale). We add a SECOND virtual queue S_n (sensing) enforcing
time-average altitude <= z_sense, mirroring the energy queue Q. FreshSky-A keeps
UAVs at the sensing altitude on average while briefly climbing the stale,
high-value ones into LoS -> attacks blockage WITHOUT abandoning sensing.
Baselines (incl. Wang'26's closed-form POMDP-Whittle index, arXiv 2605.21016)
are fixed at z_sense. Reuses the cached altitude LoS/pth cube from
joint_altitude. Run: python joint2.py [osm|proc]"""
import sys, numpy as np
import joint_altitude as J   # provides LoS[T,N,K], pth[T,N,K], ALT, N,T, PL,PN
import urban_blockage as U
LoS,pth,ALT=J.LoS,J.pth,J.ALT; N,T=LoS.shape[0]*0+J.N, LoS.shape[0]
N=J.N; ALT_arr=np.array(ALT,float); PL,PN=J.PL,J.PN
KS=1                         # z_sense = ALT[1] = 120 m (nominal sensing altitude)
Z_SENSE=ALT[KS]
ALPHA=1-1/8.0                # pi_LL retention of LoS (matches belief_whittle mean_dwell=8 -> pNL=1/8)
def wang_index(A, th, alpha, beta):
    """Wang'26 closed-form Whittle-like index W_L(delta,theta) (arXiv 2605.21016).
    Plain AoI over a Gilbert-Elliott belief channel -- value- and energy-agnostic."""
    Tt=th*alpha+(1-th)*(1-beta)
    def F1(d,t):
        den=(1-beta)*d+1-t+1e-12
        return (1-beta)/den*(d*(d+1)/2+(1-t)*beta/(1-beta)**2+(1-t)*(d+1)/(1-beta))
    def F2(d,t): return (2-t-beta)/((1-beta)*d+1-t+1e-12)
    return (F1(A+1,Tt)-F1(A,th))/(F2(A,th)-F2(A+1,Tt)+1e-12)

def run(mode, V=6.0, pbar=1.6, wev=5.0, p_event=0.08, ev_off=1/25, Amax=120, warmup=300,
        M=3, seed=0, dz_mps=12.0, rho_s=0.004, beta_ch=1/8.0):
    """mode in {fixed_map, fixed_belief, wang26, abdel, freshsky_a}. Fixed modes cruise at
    z_sense (KS). freshsky_a adapts altitude (rate-limited) with energy queue Q + sensing
    queue S enforcing avg altitude<=z_sense. Returns (wAoI, txpow/UAV, avg_alt, climb m/s)."""
    rng=np.random.default_rng(9000+seed); dz=dz_mps*U.TAU; beta=1-ALPHA
    piL=np.clip(LoS[:,:,KS].mean(0),0.05,0.95); pNL=1/8.0; pLN=np.clip(pNL*(1-piL)/piL,0,1)
    th=piL.copy(); lam=np.full(N,0.25); A=np.ones(N); Q=np.zeros(N); S=np.zeros(N)
    ev=(rng.random(N)<p_event); z=np.full(N,float(Z_SENSE))
    wA=0.0;cnt=0;esum=0.0;climb=0.0;altsum=0.0
    for t in range(T):
        w=np.where(ev,wev,1.0)
        if mode=='freshsky_a':
            # joint value of (UAV n at altitude level k): freshness gain - energy - sensing debt
            val=V*(w*A)[:,None]*LoS[t]*PL - Q[:,None]*pth[t] - rho_s*S[:,None]*np.maximum(ALT_arr[None,:]-Z_SENSE,0)
            kdes=np.argmax(val,1); d=np.clip(ALT_arr[kdes]-z,-dz,dz); z=z+d; climb+=float(np.abs(d).sum())
            ku=np.clip(np.round((z-ALT[0])/40.0).astype(int),0,len(ALT)-1)
            idx=V*w*A*LoS[t,np.arange(N),ku]*PL - Q*pth[t,np.arange(N),ku]
        else:
            ku=np.full(N,KS)
            if   mode=='fixed_map':    idx=V*w*A*LoS[t,:,KS]*PL - Q*pth[t,:,KS]     # genie at z_sense
            elif mode=='fixed_belief': idx=V*w*A*th*PL - Q*pth[t,:,KS]               # belief-Whittle
            elif mode=='wang26':       idx=np.where(pth[t,:,KS]<=U.PMAX, wang_index(A,th,ALPHA,beta), -1e18)
            elif mode=='abdel':        idx=w*A - lam*pth[t,:,KS]
        order=np.argsort(-idx); Sset=[int(n) for n in order[:M] if idx[n]>0]
        e=np.zeros(N);ack=np.zeros(N,bool);att=np.zeros(N,bool)
        for n in Sset:
            k=ku[n]; e[n]=pth[t,n,k]; att[n]=True; ack[n]=rng.random()<(PL if LoS[t,n,k] else PN)
        A=np.where(ack,1.0,np.minimum(A+1,Amax)); Q=np.maximum(Q-pbar,0.0)+e
        S=np.maximum(S+(z-Z_SENSE),0.0)                        # sensing queue (altitude-above-nominal)
        post=th.copy()
        for n in range(N):
            if att[n]:
                if ack[n]: post[n]=th[n]*PL/(th[n]*PL+(1-th[n])*PN+1e-12)
                else:      post[n]=th[n]*(1-PL)/(th[n]*(1-PL)+(1-th[n])*(1-PN)+1e-12)
        th=post*(1-pLN)+(1-post)*pNL; lam=np.maximum(lam+3e-3*(e-pbar),0.0)
        u=rng.random(N); ev=np.where(ev,u>=ev_off,u<p_event)
        if t>=warmup: wA+=float(np.sum(w*A));cnt+=1;esum+=float(np.sum(e));altsum+=float(z.mean())
    return wA/cnt, esum/cnt/N, altsum/cnt, climb/N/T/U.TAU

if __name__=="__main__":
    print(f"[{J.MODE}] z_sense={Z_SENSE}m  LoS@z_sense={LoS[:,:,KS].mean():.3f}  (alpha=piLL={ALPHA:.3f})")
    R={}
    for m,lab in [('fixed_belief','Belief-Whittle @z_s'),('wang26',"Wang'26 POMDP-Whittle @z_s"),
                  ('abdel',"Abd-Elmagid'25 @z_s"),('fixed_map','Genie @z_s (best fixed)'),
                  ('freshsky_a','FreshSky-A (joint alt)')]:
        r=np.array([run(m,seed=s) for s in range(5)]).mean(0); R[lab]=r
        print(f"  {lab:28s} wAoI={r[0]:8.1f}  pow={r[1]:.3f}mW  avg_alt={r[2]:6.1f}m  climb={r[3]:.1f}m/s")
    fa=R['FreshSky-A (joint alt)']; bw=R['Belief-Whittle @z_s']
    print(f"\n  FreshSky-A vs Belief-Whittle@z_s: AoI {100*(bw[0]-fa[0])/bw[0]:+.1f}%  (avg_alt {fa[2]:.0f} vs {bw[2]:.0f} m)")
    print("\n== FreshSky-A AoI vs its avg altitude (sweep rho_s). Compare to FIXED belief frontier:")
    print("   fixed z=80->2495, 120->1966, 160->1616, 200->1404 (from joint_altitude). If FreshSky-A")
    print("   AoI at avg_alt X is BELOW the fixed value at X, altitude REDISTRIBUTION helps. ==")
    for rho in [0.0008,0.002,0.004,0.008,0.02]:
        r=np.array([run('freshsky_a',seed=s,rho_s=rho) for s in range(4)]).mean(0)
        print(f"    FreshSky-A rho_s={rho:.4f}: avg_alt={r[2]:6.1f}m  wAoI={r[0]:8.1f}  pow={r[1]:.3f}mW  climb={r[3]:.1f}m/s")
