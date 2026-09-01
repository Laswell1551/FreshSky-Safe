# -*- coding: utf-8 -*-
"""GAMBLE step 1: does ALTITUDE control restore LoS? (make-or-break for joint
trajectory+scheduling). If a UAV can climb to clear buildings when it has fresh
high-value data, the 'unavoidable outages' that cap fixed-trajectory scheduling
become AVOIDABLE. Measure LoS fraction vs altitude on procedural + OSM geometry."""
import sys, numpy as np
import urban_blockage as U

MODE=sys.argv[1] if len(sys.argv)>1 else 'proc'
N=12; T=1200; ALT=[80,120,160,200,260,320]

if MODE=='osm':
    import osm_urban as O
    data,_=O.fetch_or_load(); polys,heights,_=O.parse_buildings(data)
    Hg,cell,sx,sy,_=O.build_grid(polys,heights); gs,_=O.place_gs(Hg,cell)
    span_x,span_y=sx,sy; tallest=heights.max()
else:
    rng=np.random.default_rng(1); Hg,cell,span=U.build_city(rng)
    gs=np.array([span*0.5,span*0.5,35.0]); span_x=span_y=span; tallest=Hg.max()
print(f"[{MODE}] grid {Hg.shape}, tallest building {tallest:.0f} m, GS z={gs[2]:.0f} m")

def nominal(seed=1):
    rng=np.random.default_rng(seed); xy=np.zeros((T,N,2))
    for n in range(N):
        p0=np.array([rng.uniform(0,span_x),rng.uniform(0,span_y)])
        ang=rng.uniform(0,2*np.pi); v=rng.uniform(6,14)*U.TAU
        vel=np.array([np.cos(ang),np.sin(ang)])*v
        for t in range(T): xy[t,n]=(p0+vel*t)%[span_x,span_y]
    return xy
xy=nominal()

LoS=np.zeros((T,N,len(ALT)),bool)
for t in range(T):
    for n in range(N):
        for k,z in enumerate(ALT):
            seg=np.sqrt((xy[t,n,0]-gs[0])**2+(xy[t,n,1]-gs[1])**2+(z-gs[2])**2)
            steps=int(np.clip(seg/(cell*0.7),60,400))
            LoS[t,n,k]=not U.los_blocked(xy[t,n,0],xy[t,n,1],z,gs[0],gs[1],gs[2],Hg,cell,span_x,steps=steps)

print("LoS fraction vs altitude:")
for k,z in enumerate(ALT): print(f"   z={z:3d} m : LoS={LoS[:,:,k].mean():.3f}")
print(f"LoS at nominal 80-120 m band (k=0..1 avg): {LoS[:,:,:2].mean():.3f}")
print(f"LoS if free to pick BEST altitude per slot: {LoS.any(2).mean():.3f}")
blocked_low=~LoS[:,:,0]; rescued=blocked_low & LoS.any(2)
print(f"of slots blocked at 80 m, fraction rescued by climbing: {rescued.sum()/max(blocked_low.sum(),1):.3f}")

# distances + delivery power per altitude
D=np.zeros((T,N,len(ALT)))
for k,z in enumerate(ALT):
    D[:,:,k]=np.sqrt((xy[:,:,0]-gs[0])**2+(xy[:,:,1]-gs[1])**2+(z-gs[2])**2)
fspl=20*np.log10(4*np.pi*U.FC*D/3e8)
pth=np.minimum(U.GAMMA*U.SIGMA2/10**(-(fspl+U.ETAL)/10), U.PMAX)   # [T,N,K] LoS-case power
PL,PN=U.PHI_LOS,U.PHI_NLOS

ALT_arr=np.array(ALT,float)
def run(mode, V=6.0, pbar=1.6, wev=5.0, p_event=0.08, ev_off=1/25, Amax=120,
        warmup=300, M=3, seed=0, fk=1, dz_mps=6.0, climb_wpm=0.0):
    """mode: 'fixed' belief-Whittle cruising at altitude level fk; 'joint' picks
    best altitude per UAV each slot with a REALISTIC climb-rate limit dz_mps
    (m/s) and a per-metre climb penalty climb_wpm (discourages needless climbing).
    Deliverability/energy use the UAV's ACTUAL current altitude."""
    rng=np.random.default_rng(9000+seed); dz_slot=dz_mps*U.TAU     # metres per slot
    piL=np.clip(LoS[:,:,fk].mean(0),0.05,0.95); pNL=1/8.0; pLN=np.clip(pNL*(1-piL)/piL,0,1)
    th=piL.copy(); A=np.ones(N); Q=np.zeros(N); ev=(rng.random(N)<p_event)
    z=np.full(N,float(ALT[fk]))                                    # actual altitude (m), continuous
    wA=0.0;cnt=0;esum=0.0; climb=0.0
    for t in range(T):
        w=np.where(ev,wev,1.0)
        if mode=='joint':
            idxk=V*(w*A)[:,None]*LoS[t]*PL - Q[:,None]*pth[t] - climb_wpm*np.abs(ALT_arr[None,:]-z[:,None])
            kdes=np.argmax(idxk,1); ztgt=ALT_arr[kdes]
            dz=np.clip(ztgt-z,-dz_slot,dz_slot); z=z+dz; climb+=float(np.sum(np.abs(dz)))
            kuse=np.clip(np.round((z-ALT[0])/40.0).astype(int),0,len(ALT)-1)   # nearest grid level
            idx=V*w*A*LoS[t,np.arange(N),kuse]*PL - Q*pth[t,np.arange(N),kuse]
        else:
            kuse=np.full(N,fk); idx=V*w*A*th*PL - Q*pth[t,:,fk]
        order=np.argsort(-idx); S=[int(n) for n in order[:M] if idx[n]>0]
        e=np.zeros(N);ack=np.zeros(N,bool);att=np.zeros(N,bool)
        for n in S:
            k=kuse[n]; e[n]=pth[t,n,k]; att[n]=True
            ack[n]=rng.random()<(PL if LoS[t,n,k] else PN)
        A=np.where(ack,1.0,np.minimum(A+1,Amax)); Q=np.maximum(Q-pbar,0.0)+e
        post=th.copy()
        for n in range(N):
            if att[n]:
                if ack[n]: post[n]=th[n]*PL/(th[n]*PL+(1-th[n])*PN+1e-12)
                else:      post[n]=th[n]*(1-PL)/(th[n]*(1-PL)+(1-th[n])*(1-PN)+1e-12)
        th=post*(1-pLN)+(1-post)*pNL
        u=rng.random(N); ev=np.where(ev,u>=ev_off,u<p_event)
        if t>=warmup: wA+=float(np.sum(w*A));cnt+=1;esum+=float(np.sum(e))
    return wA/cnt, esum/cnt/N, climb/N/T     # (wAoI, tx power/UAV, mean |climb| m/slot/UAV)

print("\n== fixed cruising altitude sweep (belief-Whittle) -- STATIC benefit ==")
for fk,z in enumerate(ALT):
    r=np.mean([run('fixed',fk=fk,seed=s) for s in range(4)],axis=0)
    print(f"  fixed z={z:3d} m : wAoI={r[0]:8.1f}  pow/UAV={r[1]:.3f} mW")
print("== joint altitude+scheduling (realistic climb-rate limit; rho=propulsion penalty) ==")
for dz,rho,lab in [(1000.,0.0,'idealized'),(12.,0.0,'12 m/s'),(6.,0.0,'6 m/s'),
                   (12.,0.02,'12 m/s + propulsion pen.'),(6.,0.02,'6 m/s + propulsion pen.')]:
    r=np.mean([run('joint',fk=1,seed=s,dz_mps=dz,climb_wpm=rho) for s in range(4)],axis=0)
    print(f"  joint [{lab:26s}] wAoI={r[0]:8.1f}  pow/UAV={r[1]:.3f} mW  climb={r[2]:.2f} m/slot/UAV ({r[2]*10:.1f} m/s avg)")
