# -*- coding: utf-8 -*-
"""Belief-Whittle on the SAME air-ground channel model as the main sim
(Al-Hourani LoS prob, distance-dependent p_th energy), under PARTIAL
observability: the LoS/NLoS state is a hidden Gilbert-Elliott chain observed
only via ACK/NACK, with a proper Bayesian belief. Belief-Whittle exploits the
correlation to Pareto-dominate memoryless scheduling; the gain grows with
correlation and vanishes at i.i.d. Self-contained (numpy), reuses main-sim
channel constants."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
FIG=os.path.join(os.path.dirname(__file__),"figs")
RES=os.path.join(os.path.dirname(__file__),"results")

# --- air-ground channel constants (identical to freshsky_sim.Cfg) ---
H=100.0; A1=9.61; A2=0.16; ETAL=1.0; ETAN=20.0; FC=2.4e9; W=20e6; TAU=0.1
PMAX=100.0; NF=10**(0.7); DBITS=4.0e6
SIGMA2=1.38e-23*290.0*W*NF*1e3               # mW
GAMMA=2**(DBITS/(W*TAU))-1
PHI_LOS=0.90; PHI_NLOS=0.02                   # deliver prob given LoS / NLoS at p_th_los

def plos(th_deg): return 1.0/(1.0+A1*np.exp(-A2*(th_deg-A1)))

class Cfg:
    N=12; M=3; T=4000; warmup=300
    mean_dwell=8.0                            # mean NLoS dwell (correlation knob)
    wev=5.0; p_event=0.08; ev_off=1/25
    Amax=120; V=6.0; pbar=1.6                  # per-UAV avg power budget (mW)

def geometry(cfg, rng):
    r=rng.uniform(60,380,cfg.N); d=np.sqrt(H**2+r**2)
    th=np.degrees(np.arctan(H/r)); piL=np.clip(plos(th),0.05,0.95)
    fspl=20*np.log10(4*np.pi*FC*d/3e8)
    Llos=10**(-(fspl+ETAL)/10); pth_los=GAMMA*SIGMA2/Llos      # mW to deliver if LoS
    return piL, np.minimum(pth_los,PMAX)

def run(policy, cfg, seeds=(0,1,2,3,4), genie=False, details=False):
    ws=[]; es=[]; mx=[]
    for sd in seeds:
        rng=np.random.default_rng(7000+sd); N=cfg.N
        pol=policy(cfg) if isinstance(policy,type) else policy
        if hasattr(pol,'reset_seed'):
            pol.reset_seed(sd)
        piL,pth=geometry(cfg,rng)
        pNL=1.0/cfg.mean_dwell; pLN=np.clip(pNL*(1-piL)/piL,0,1)   # GE, stationary=piL
        globals()['_ALPHA']=1.0-pLN; globals()['_BETA']=1.0-pNL    # GE retention probs (for Wang'26 index)
        los=(rng.random(N)<piL)                                    # hidden state
        th=piL.copy()                                              # belief P(LoS)
        A=np.ones(N); Q=np.zeros(N); ev=(rng.random(N)<cfg.p_event)
        wA=0.0; cnt=0; esum=0.0; esum_n=np.zeros(N)
        for t in range(cfg.T):
            w=np.where(ev,cfg.wev,1.0)
            bel=los.astype(float) if genie else th
            S=pol(A,bel,Q,w,pth,cfg)
            e=np.zeros(N); ack=np.zeros(N,bool); attempted=np.zeros(N,bool)
            for n in S:
                e[n]=pth[n]; attempted[n]=True
                p=PHI_LOS if los[n] else PHI_NLOS
                ack[n]=(rng.random()<p)
            if hasattr(pol,'update'): pol.update(e)
            if hasattr(pol,'observe_feedback'):
                pol.observe_feedback(attempted,ack)
            A=np.where(ack,1.0,np.minimum(A+1,cfg.Amax))
            Q=np.maximum(Q-cfg.pbar,0.0)+e
            # Bayesian belief update then propagate through GE chain
            post=th.copy()
            for n in range(N):
                if attempted[n]:
                    if ack[n]: post[n]=th[n]*PHI_LOS/(th[n]*PHI_LOS+(1-th[n])*PHI_NLOS+1e-12)
                    else:      post[n]=th[n]*(1-PHI_LOS)/(th[n]*(1-PHI_LOS)+(1-th[n])*(1-PHI_NLOS)+1e-12)
            th=post*(1-pLN)+(1-post)*pNL                          # propagate P(LoS next)
            # hidden state + events transition
            u=rng.random(N); los=np.where(los,u>=pLN,u<pNL)
            u=rng.random(N); ev=np.where(ev,u>=cfg.ev_off,u<cfg.p_event)
            if t>=cfg.warmup:
                wA+=float(np.sum(w*A)); cnt+=1; esum+=float(np.sum(e)); esum_n+=e
        ws.append(wA/cnt); es.append(esum/cnt/N); mx.append(float(np.max(esum_n/cnt)))
    if details:
        return dict(wAoI=float(np.mean(ws)),wAoI_std=float(np.std(ws,ddof=1)),
                    power=float(np.mean(es)),max_power=float(np.mean(mx)),
                    feasible=bool(np.all(np.asarray(mx)<=cfg.pbar*1.02)))
    return np.mean(ws), np.mean(es)

def topM(score,cfg):
    idx=np.argsort(-score); return [int(n) for n in idx[:cfg.M] if score[n]>0]
def belief_whittle(A,bel,Q,w,pth,cfg): return topM(cfg.V*w*A*bel*PHI_LOS - Q*pth, cfg)  # ours
def memoryless(A,bel,Q,w,pth,cfg):
    return topM(cfg.V*w*A - Q*pth, cfg)                                                 # ignores belief

class BeliefDPP26:
    """Known-transition belief-DPP oracle adapted from Zhou et al. (2026).
    It retains their single aggregate budget queue; using the true
    transition law makes it at least as informed as the Thompson-sampling
    version, while exposing the distinction from FreshSky's per-UAV queues."""
    def __init__(self,cfg): self.cfg=cfg; self.Z=0.0
    def __call__(self,A,bel,Q,w,pth,cfg):
        score=cfg.V*w*A*bel*PHI_LOS-self.Z*pth
        return topM(score,cfg)
    def update(self,e):
        self.Z=max(0.0,self.Z+float(np.sum(e))-self.cfg.N*self.cfg.pbar)

class TSDPPAdapted26:
    """ACK/NACK-channel adaptation of Zhou et al.'s 2026 TS-DPP.

    The published method learns unknown event-state transitions and uses one
    aggregate budget queue. Here each hidden Gilbert--Elliott link is an arm:
    Beta posteriors learn its one-step LoS-retention and NLoS-to-LoS
    probabilities from consecutive ACK/NACK observations. The policy samples
    transitions once per episode, propagates its own belief, and never receives
    the simulator's true transition law. ACK/NACK is a noisy state indicator,
    so this is deliberately labelled an adaptation rather than an exact
    reproduction of the event-capture algorithm.
    """
    EPISODE=100
    def __init__(self,cfg):
        self.cfg=cfg
        self.reset_seed(0)

    def reset_seed(self,seed):
        N=self.cfg.N
        self.rng=np.random.default_rng(91000+int(seed))
        self.Z=0.0
        self.t=0
        self.th=np.full(N,0.5)
        # Beta(success, failure) for p_LL and p_NL, respectively.
        self.aLL=np.ones(N); self.bLL=np.ones(N)
        self.aNL=np.ones(N); self.bNL=np.ones(N)
        self.prev_state=np.full(N,-1,dtype=int)
        self.prev_t=np.full(N,-2,dtype=int)
        self._resample()

    def _resample(self):
        self.pLL=self.rng.beta(self.aLL,self.bLL)
        self.pNL=self.rng.beta(self.aNL,self.bNL)
        # Positive correlation is the relevant air-ground regime. Clipping
        # prevents a diffuse early posterior from creating degenerate beliefs.
        self.pLL=np.clip(self.pLL,0.50,0.999)
        self.pNL=np.clip(self.pNL,0.001,0.50)

    def __call__(self,A,bel,Q,w,pth,cfg):
        del bel,Q  # no oracle belief or individual queue is available
        score=cfg.V*w*A*self.th*PHI_LOS-self.Z*pth
        return topM(score,cfg)

    def update(self,e):
        self.Z=max(0.0,self.Z+float(np.sum(e))-self.cfg.N*self.cfg.pbar)

    def observe_feedback(self,attempted,ack):
        attempted=np.asarray(attempted,dtype=bool)
        ack=np.asarray(ack,dtype=bool)
        post=self.th.copy()
        for n in np.flatnonzero(attempted):
            # Exact Bayesian observation update for noisy ACK/NACK.
            if ack[n]:
                post[n]=self.th[n]*PHI_LOS/(self.th[n]*PHI_LOS+
                    (1-self.th[n])*PHI_NLOS+1e-12)
                state=1
            else:
                post[n]=self.th[n]*(1-PHI_LOS)/(self.th[n]*(1-PHI_LOS)+
                    (1-self.th[n])*(1-PHI_NLOS)+1e-12)
                state=0
            # A one-step transition is identifiable only for consecutive
            # observations of the same arm.
            if self.prev_t[n]==self.t-1:
                prev=self.prev_state[n]
                if prev==1:
                    if state==1: self.aLL[n]+=1
                    else:        self.bLL[n]+=1
                else:
                    if state==1: self.aNL[n]+=1
                    else:        self.bNL[n]+=1
            self.prev_state[n]=state
            self.prev_t[n]=self.t
        self.th=post*self.pLL+(1-post)*self.pNL
        self.t+=1
        if self.t%self.EPISODE==0:
            self._resample()

def wang_index(A, th, alpha, beta):
    """Wang et al. 2026 (arXiv 2605.21016) closed-form POMDP Whittle-like index
    W_L(delta,theta) for age-optimal scheduling over a Gilbert-Elliott belief
    channel. PLAIN AoI: no importance value, no energy price."""
    Tt=th*alpha+(1-th)*(1-beta)
    def F1(d,t):
        den=(1-beta)*d+1-t+1e-12
        return (1-beta)/den*(d*(d+1)/2+(1-t)*beta/(1-beta)**2+(1-t)*(d+1)/(1-beta))
    def F2(d,t): return (2-t-beta)/((1-beta)*d+1-t+1e-12)
    return (F1(A+1,Tt)-F1(A,th))/(F2(A,th)-F2(A+1,Tt)+1e-12)
def wang26(A,bel,Q,w,pth,cfg):                                # Wang'26 POMDP-Whittle (value-/energy-agnostic)
    return topM(wang_index(A, bel, _ALPHA, _BETA), cfg)
def clone(cfg,**kw):
    c=Cfg()
    for k,v in kw.items(): setattr(c,k,v)
    return c

if __name__=="__main__":
    cfg=Cfg()
    print(f"air-ground: gamma={GAMMA:.2f} sigma2={10*np.log10(SIGMA2):.0f}dBm budget={cfg.pbar}mW")
    print("== default (partial observability) ==")
    rows=[]
    for name,pol,g in [("Genie (knows LoS)",belief_whittle,True),
                       ("Belief-Whittle (ours)",belief_whittle,False),
                       ("Belief-DPP (Zhou'26)",BeliefDPP26,False),
                       ("TS-DPP adapted (Zhou'26)",TSDPPAdapted26,False),
                       ("Memoryless (no belief)",memoryless,False)]:
        d=run(pol,cfg,genie=g,details=True); rows.append(dict(policy=name,**d))
        print(f"  {name:24s} wAoI={d['wAoI']:7.1f}  power/UAV={d['power']:.3f}mW  "
              f"max={d['max_power']:.3f} feasible={d['feasible']}")
    os.makedirs(RES,exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES,"belief_recent.csv"),index=False)
    tight=clone(cfg,pbar=0.4); tight_rows=[]
    for name,pol in [("FreshSky-B (ours)",belief_whittle),
                     ("Belief-DPP (Zhou'26)",BeliefDPP26),
                     ("TS-DPP adapted (Zhou'26)",TSDPPAdapted26),
                     ("Memoryless",memoryless)]:
        tight_rows.append(dict(policy=name,**run(pol,tight,details=True)))
    pd.DataFrame(tight_rows).to_csv(os.path.join(RES,"belief_recent_tight.csv"),index=False)
    print("== correlation sweep ==")
    dwell=[1.5,2,3,5,8,14,20]; BW=[];ML=[];gain=[]
    for md in dwell:
        c=clone(cfg,mean_dwell=md); bw,_=run(belief_whittle,c); ml,_=run(memoryless,c)
        BW.append(bw);ML.append(ml);gain.append(100*(ml-bw)/ml)
        print(f"  mean NLoS dwell={md:4.1f}  BW={bw:6.1f} ML={ml:6.1f} gain={gain[-1]:5.1f}%")
    fig,ax=plt.subplots(figsize=(5.2,3.2)); ax.plot(dwell,gain,'-o',color='#2ca02c',lw=1.8); ax.axhline(0,ls=':',color='0.6')
    ax.set_xlabel("mean NLoS dwell (slots) = correlation"); ax.set_ylabel("Belief-Whittle AoI gain (\\%)")
    ax.set_title("Gain grows with correlation (air-ground channel)")
    fig.savefig(os.path.join(FIG,"e10_belief_sweep.png"),dpi=150,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.2,3.2))
    for name,pol,g,mk,cl in [("Genie",belief_whittle,True,'*','0.4'),("Belief-Whittle (ours)",belief_whittle,False,'o','#2ca02c'),
                             ("Belief-DPP'26 oracle",BeliefDPP26,False,'^','#9467bd'),
                             ("TS-DPP'26 adapted",TSDPPAdapted26,False,'D','#e377c2'),
                             ("Memoryless",memoryless,False,'s','#1f77b4')]:
        wA,en=run(pol,cfg,genie=g); ax.scatter(en,wA,marker=mk,s=110,color=cl,label=name,zorder=3)
    ax.set_xlabel("power per UAV (mW)"); ax.set_ylabel("weighted AoI"); ax.legend(fontsize=8)
    ax.set_title("Belief-Whittle Pareto-dominates (air-ground)")
    fig.savefig(os.path.join(FIG,"e11_belief_pareto.png"),dpi=150,bbox_inches='tight'); plt.close(fig)
    print("saved e10_belief_sweep.png, e11_belief_pareto.png")
