# -*- coding: utf-8 -*-
"""
FreshSky simulation harness.
Value-aware age-optimal uplink scheduling for multi-UAV low-altitude sensing
over intermittent (correlated-blockage) air-ground channels.

Implements the FreshSky scheduling model and a rich set of baselines, each
mapped to a published reference (see BASELINES dict docstrings).
"""
import os, time, json, math
import numpy as np
import pandas as pd
from scipy.optimize import linprog

RNG_MASTER = 20270731  # deterministic

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
class Cfg:
    # network
    N = 12                 # UAVs / sources
    M = 3                  # orthogonal uplink sub-bands (M < N)
    T = 3000               # slots per episode
    tau = 0.1              # slot length (s)  -> 100 ms
    # PHY (2.4 GHz WiFi-like) -- ALL POWERS IN mW so value & energy terms are comparable
    fc = 2.4e9
    W = 20e6               # per-sub-band bandwidth (Hz)
    Pmax = 100.0           # max Tx power (mW) = 20 dBm
    pbar_frac = 0.03       # avg-power budget as fraction of Pmax  -> tuned to bind
    Dbits = 4.0e6          # update size (bits)  -> gamma = 2^(D/(W tau)) - 1
    # air-ground channel (Al-Hourani urban)
    H = 100.0              # UAV height (m)
    rmin, rmax = 50.0, 400.0
    a1, a2 = 9.61, 0.16    # Al-Hourani urban LoS-prob constants
    etaL_dB, etaN_dB = 1.0, 20.0   # excess path loss LoS / NLoS (dB)
    mL, mN = 3.0, 1.0      # Nakagami-m LoS / NLoS
    # blockage temporal correlation (mean sojourn in slots)
    mean_sojourn_N = 8.0   # mean NLoS dwell (correlation) -> bursty outages
    # interference: per-link 2-state (low/high) Markov, in mW
    I_low, I_high = 0.0, 8e-9        # mW (high ~ -81 dBm >> noise)
    p_ihigh, mean_sojourn_I = 0.30, 6.0
    # value / event process: w in {1 (normal), wev (event)}, events markovian
    wev = 10.0
    p_event, mean_sojourn_ev = 0.08, 25.0
    Amax = 500
    # Lyapunov knob (power in mW -> V ~ O(1))
    V = 2.0
    # noise
    NF_dB = 7.0
    seed = 0

    def derived(self):
        k, T0 = 1.38e-23, 290.0
        self.sigma2 = k*T0*self.W*(10**(self.NF_dB/10.0))*1e3   # noise power (mW)
        self.gamma = 2**(self.Dbits/(self.W*self.tau)) - 1.0
        self.pbar = self.pbar_frac*self.Pmax
        return self

def clone(cfg, **kw):
    c = Cfg()
    for a in dir(cfg):
        if not a.startswith('_') and not callable(getattr(cfg,a)):
            setattr(c,a,getattr(cfg,a))
    for k,v in kw.items(): setattr(c,k,v)
    return c.derived()

# --------------------------------------------------------------------------
# Channel helpers
# --------------------------------------------------------------------------
def plos(theta_deg, a1, a2):
    return 1.0/(1.0 + a1*np.exp(-a2*(theta_deg - a1)))

def fspl_dB(d, fc):
    c = 3e8
    return 20*np.log10(4*np.pi*fc*d/c)

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
class Env:
    def __init__(self, cfg):
        self.cfg = cfg.derived() if not hasattr(cfg,'sigma2') else cfg
        self.rng = np.random.default_rng(RNG_MASTER + 1000*cfg.seed)
        c = self.cfg; N = c.N
        # patrol: each UAV oscillates horizontal distance sinusoidally
        self.phase = self.rng.uniform(0, 2*np.pi, N)
        self.freq  = self.rng.uniform(0.002, 0.02, N)   # slow mobility
        # blockage state (0=LoS,1=NLoS), interference (0=low,1=high), event
        self.beta = self.rng.integers(0,2,N)
        self.iis  = (self.rng.random(N) < c.p_ihigh).astype(int)
        self.ev   = (self.rng.random(N) < c.p_event).astype(int)
        self.A = np.ones(N)          # AoI
        self.Q = np.zeros(N)         # virtual energy queue
        self.t = 0
        # for statistical phi(beta): success prob per (uav,beta) estimated online
        self.phi_hits = np.ones((N,2)); self.phi_try = np.ones((N,2))

    def _rh(self, n):
        c=self.cfg
        return c.rmin + 0.5*(c.rmax-c.rmin)*(1+np.sin(2*np.pi*self.freq[n]*self.t + self.phase[n]))

    def large_scale(self):
        """Return per-UAV large-scale gain L (linear), theta, plos, d."""
        c=self.cfg; N=c.N
        rh = np.array([self._rh(n) for n in range(N)])
        d = np.sqrt(c.H**2 + rh**2)
        theta = np.degrees(np.arctan(c.H/rh))
        eta = np.where(self.beta==0, c.etaL_dB, c.etaN_dB)
        L_dB = fspl_dB(d, c.fc) + eta
        L = 10**(-L_dB/10.0)
        return L, theta, d

    def channel_gain(self, L):
        """Instantaneous gain incl. Nakagami fading (known to full-CSI schedulers)."""
        c=self.cfg; N=c.N
        m = np.where(self.beta==0, c.mL, c.mN)
        # Nakagami power |h|^2 ~ Gamma(shape=m, scale=1/m), mean 1
        fad = self.rng.gamma(shape=m, scale=1.0/m)
        return L*fad

    def interference(self):
        c=self.cfg
        return np.where(self.iis==0, c.I_low, c.I_high)

    def weights(self):
        c=self.cfg
        return np.where(self.ev==0, 1.0, c.wev)

    def p_th(self, g, I):
        c=self.cfg
        return c.gamma*(c.sigma2 + I)/np.maximum(g,1e-30)

    def phi_beta(self):
        """Statistical success prob per UAV given current beta (for Whittle)."""
        return self.phi_hits[np.arange(self.cfg.N), self.beta]/self.phi_try[np.arange(self.cfg.N), self.beta]

    def observe(self):
        L, theta, d = self.large_scale()
        g = self.channel_gain(L)
        I = self.interference()
        pth = self.p_th(g, I)
        deliverable = (pth <= self.cfg.Pmax)
        return dict(L=L, g=g, I=I, pth=pth, deliverable=deliverable,
                    A=self.A.copy(), Q=self.Q.copy(), w=self.weights(),
                    beta=self.beta.copy(), theta=theta)

    def step(self, sched, power):
        """sched: index array of scheduled UAVs (<=M); power: per-UAV Tx power (0 if not sched)."""
        c=self.cfg; N=c.N
        obs = self._last
        s = np.zeros(N, dtype=int)     # delivery indicator
        for n in sched:
            # delivery deterministic given known g & power (full CSI)
            deliver = (power[n]*obs['g'][n]/(c.sigma2+obs['I'][n])) >= c.gamma - 1e-12
            s[n] = 1 if deliver else 0
            # update phi estimate
            self.phi_try[n, self.beta[n]] += 1
            self.phi_hits[n, self.beta[n]] += s[n]
        # AoI update
        self.A = np.where(s==1, 1.0, np.minimum(self.A+1, c.Amax))
        # energy virtual queue
        p = np.array([power[n] for n in range(N)])
        self.Q = np.maximum(self.Q - c.pbar, 0.0) + p
        # exogenous dynamics: blockage / interference / event Markov transitions
        self._advance_markov()
        self.t += 1
        return s, p

    def _advance_markov(self):
        c=self.cfg; N=c.N
        # blockage: stationary LoS prob from Al-Hourani at current theta
        _, theta, _ = self.large_scale()
        piL = plos(theta, c.a1, c.a2)                      # target stationary LoS prob
        pNL = 1.0/c.mean_sojourn_N                         # NLoS->LoS
        pLN = np.clip(pNL*(1-piL)/np.maximum(piL,1e-3), 0, 1)  # LoS->NLoS
        u = self.rng.random(N)
        newbeta = self.beta.copy()
        los = (self.beta==0); nlos=~los
        newbeta[los]  = np.where(u[los]  < pLN[los], 1, 0)
        newbeta[nlos] = np.where(u[nlos] < pNL,      0, 1)
        self.beta = newbeta
        # interference 2-state
        pIH = c.p_ihigh; pI_HL = 1.0/c.mean_sojourn_I
        pI_LH = np.clip(pI_HL*pIH/max(1-pIH,1e-3),0,1)
        u=self.rng.random(N); low=(self.iis==0)
        self.iis = np.where(low, (u<pI_LH).astype(int), (u>=pI_HL).astype(int))
        # event process
        pE=c.p_event; pE_off=1.0/c.mean_sojourn_ev
        pE_on=np.clip(pE_off*pE/max(1-pE,1e-3),0,1)
        u=self.rng.random(N); noe=(self.ev==0)
        self.ev = np.where(noe, (u<pE_on).astype(int), (u>=pE_off).astype(int))

# --------------------------------------------------------------------------
# Policies  (each returns (scheduled_list, power_dict))
# --------------------------------------------------------------------------
def _assign(order, obs, cfg, power_fn):
    """Take UAVs in priority order; grant up to M that are deliverable & positive."""
    sched=[]; power={n:0.0 for n in range(cfg.N)}
    for n in order:
        if len(sched)>=cfg.M: break
        if not obs['deliverable'][n]: continue
        p = power_fn(n)
        if p<=0 or p>cfg.Pmax+1e-12: continue
        sched.append(n); power[n]=p
    return sched, power

class Policy:
    name="base"; ref=""
    def __init__(self,cfg): self.cfg=cfg
    def decide(self, env, obs): raise NotImplementedError

class RoundRobin(Policy):
    name="Round-Robin"; ref="fairness baseline"
    def __init__(self,cfg): super().__init__(cfg); self.ptr=0
    def decide(self, env, obs):
        order=[(self.ptr+i)%self.cfg.N for i in range(self.cfg.N)]
        self.ptr=(self.ptr+self.cfg.M)%self.cfg.N
        return _assign(order,obs,self.cfg,lambda n:obs['pth'][n])

class MaxAgeFirst(Policy):
    name="Max-Age-First"; ref="Kadota-Modiano, ToN'18 (greedy AoI)"
    def decide(self, env, obs):
        order=np.argsort(-obs['A'])
        return _assign(order,obs,self.cfg,lambda n:obs['pth'][n])

class MaxWeight(Policy):
    name="Max-Weight"; ref="Kadota-Modiano, ToN'18 (w*A, no energy)"
    def decide(self, env, obs):
        score=obs['w']*obs['A']
        order=np.argsort(-score)
        return _assign(order,obs,self.cfg,lambda n:obs['pth'][n])

class WhittleAoI(Policy):
    name="Whittle-AoI"; ref="Tripathi-Modiano, ToN'19 (index, no value/energy)"
    def decide(self, env, obs):
        # classic AoI Whittle index ~ A*(A+2)*p (here success prob ~1 if deliverable)
        idx=obs['A']*(obs['A']+2.0)
        order=np.argsort(-idx)
        return _assign(order,obs,self.cfg,lambda n:obs['pth'][n])

class Opportunistic(Policy):
    name="Opportunistic"; ref="max-SINR / best-channel (channel-aware)"
    def decide(self, env, obs):
        order=np.argsort(obs['pth'])   # smallest required power first
        return _assign(order,obs,self.cfg,lambda n:obs['pth'][n])

class VoIGreedy(Policy):
    name="VoI-Greedy"; ref="Maatouk et al., ToN'20 (AoII / value-greedy)"
    def decide(self, env, obs):
        score=obs['w']*obs['A']
        order=np.argsort(-score)   # value-weighted but ignores energy & channel cost
        return _assign(order,obs,self.cfg,lambda n:obs['pth'][n])

class DPP_NoValue(Policy):
    name="Lyap-DPP (no value)"; ref="He et al. (2024) / Neely (2010), energy-aware and unweighted"
    def decide(self, env, obs):
        c=self.cfg
        R=np.where(obs['deliverable'], c.V*obs['A']-obs['Q']*obs['pth'], -1e18)
        order=np.argsort(-R)
        return _assign([n for n in order if R[n]>0],obs,c,lambda n:obs['pth'][n])

class FreshSkyDPP(Policy):
    name="FreshSky-DPP (ours)"; ref="ours"
    def decide(self, env, obs):
        c=self.cfg
        R=np.where(obs['deliverable'], c.V*obs['w']*obs['A']-obs['Q']*obs['pth'], -1e18)
        order=np.argsort(-R)
        return _assign([n for n in order if R[n]>0],obs,c,lambda n:obs['pth'][n])

class FreshSkyWhittle(Policy):
    name="FreshSky-Whittle (ours)"; ref="ours (statistical index, Thm 3)"
    def decide(self, env, obs):
        c=self.cfg
        phi=env.phi_beta()   # expected success prob given beta
        W=np.where(obs['deliverable'], c.V*obs['w']*obs['A']*phi-obs['Q']*obs['pth'], -1e18)
        order=np.argsort(-W)
        return _assign([n for n in order if W[n]>0],obs,c,lambda n:obs['pth'][n])

class FreshSkyPred(Policy):
    name="FreshSky-P (ours)"; ref="ours + Markov-blockage lookahead (exploits correlation)"
    KAPPA=1.0    # lookahead weight
    def decide(self, env, obs):
        c=self.cfg
        # Markov blockage lookahead: a currently-LoS (cheap) link may go NLoS and
        # then stay blocked ~mean_sojourn_N slots -> serve it now before the outage.
        piL=np.clip(plos(obs['theta'],c.a1,c.a2),1e-3,1-1e-3)   # stationary LoS prob
        pNL=1.0/c.mean_sojourn_N
        pLN=np.clip(pNL*(1-piL)/piL,0,1)                        # P(LoS->NLoS)
        los=(obs['beta']==0).astype(float)
        anticip=self.KAPPA*los*pLN*c.mean_sojourn_N            # expected age saved by serving now
        R=np.where(obs['deliverable'],
                   c.V*obs['w']*(obs['A']+anticip)-obs['Q']*obs['pth'], -1e18)
        order=np.argsort(-R)
        return _assign([n for n in order if R[n]>0],obs,c,lambda n:obs['pth'][n])

class AoIIWhittle(Policy):
    name="AoII-Whittle"; ref="Kriouile-Assaad, ISIT'21 (value-aware AoII index, no energy)"
    def decide(self, env, obs):
        idx=obs['w']*obs['A']*(obs['A']+2.0)      # value-weighted quadratic (Whittle) index
        order=np.argsort(-idx)
        return _assign(order,obs,self.cfg,lambda n:obs['pth'][n])

class CostWhittle(Policy):
    name="Cost-AoI-Whittle"; ref="Tang et al., CommL'22 (energy-priced AoI index, no value)"
    def decide(self, env, obs):
        c=self.cfg
        idx=np.where(obs['deliverable'], obs['A']*(obs['A']+2.0)-obs['Q']*obs['pth'], -1e18)
        order=np.argsort(-idx)
        return _assign([n for n in order if idx[n]>0],obs,c,lambda n:obs['pth'][n])

class ZhouLinCW(Policy):
    name="CW-Whittle [Zhou-Lin'24]"; ref="Zhou-Lin, MobiHoc'24 (channel-weighted quadratic value index, no energy)"
    def decide(self, env, obs):
        phi=env.phi_beta()
        idx=obs['w']*phi*obs['A']*(obs['A']+1.0)/2.0   # w_n * p_n * A(A+1)/2
        order=np.argsort(-idx)
        return _assign(order,obs,self.cfg,lambda n:obs['pth'][n])

class FixedPriceThr(Policy):
    """Static-price AoI-energy threshold: idx = w*A - LAM*p_th with a SINGLE
    energy price LAM fixed offline. A legitimate, widely-used baseline; it is
    feasible only for the regime its price was tuned on and cannot react to a
    distribution shift (that is the point of the shift experiment)."""
    name="Fixed-Price Thr."; ref="static-lambda Lagrangian AoI-energy threshold (price tuned offline)"
    LAM=0.25    # offline-tuned energy price
    def decide(self, env, obs):
        c=self.cfg
        idx=np.where(obs['deliverable'], obs['w']*obs['A'] - self.LAM*obs['pth'], -1e18)
        order=np.argsort(-idx)
        return _assign([n for n in order if idx[n]>0],obs,c,lambda n:obs['pth'][n])

class AoIEnergyOnline(Policy):
    """Adaptation of Abd-Elmagid et al. (MobiHoc 2025): the energy price is
    learned online by per-UAV stochastic dual ascent.  Their finite-time regret
    result concerns a soft AoI--energy objective; this adaptation is not a hard
    per-UAV budget certificate under distribution shift."""
    name="AoI-Energy Online [Abd-Elmagid'25]"
    ref="Abd-Elmagid et al., MobiHoc'25 (online-learned energy price, unknown statistics)"
    ETA=3e-3    # dual step size (the learning rate the method must set)
    LAM0=0.25
    def __init__(self,cfg):
        super().__init__(cfg); self.lam=np.full(cfg.N, self.LAM0)  # per-UAV price, learned online
    def decide(self, env, obs):
        c=self.cfg
        idx=np.where(obs['deliverable'], obs['w']*obs['A'] - self.lam*obs['pth'], -1e18)
        order=np.argsort(-idx)
        sched,power=_assign([n for n in order if idx[n]>0],obs,c,lambda n:obs['pth'][n])
        p=np.array([power[n] for n in range(c.N)])
        self.lam=np.maximum(self.lam + self.ETA*(p - c.pbar), 0.0)   # online dual ascent
        return sched,power

class TangCMDP(Policy):
    """Truncated constrained-Markov-decision-process policy of Tang et al.
    (IEEE JSAC 2020), adapted to the present air--ground channel.

    Required power is quantized into four deliverable channel states plus one
    blocked state.  A finite-state occupancy-measure linear program is solved
    offline for every UAV.  A common bandwidth multiplier is found by bisection,
    and the two bracketing stationary policies are mixed exactly as in the
    relaxed construction.  At run time, requesting UAVs are truncated uniformly
    at random to the M available sub-bands.  The baseline uses plain AoI, as the
    published method does, and never observes future samples or the test trace.
    """
    name="Tang-CMDP [JSAC'20]"
    ref="Tang et al., IEEE JSAC 2020 (offline CMDP/LP with random truncation)"
    XMAX=40
    QBINS=4
    TRAIN_SLOTS=6000
    _CACHE={}

    def __init__(self,cfg):
        super().__init__(cfg)
        self.rng=np.random.default_rng(RNG_MASTER+23000+cfg.seed)
        key=(cfg.seed,cfg.N,cfg.M,cfg.pbar,cfg.Pmax,cfg.H,cfg.rmin,cfg.rmax,
             cfg.mean_sojourn_N,cfg.p_ihigh,cfg.mean_sojourn_I,cfg.I_high,
             cfg.Dbits,cfg.W,cfg.tau)
        if key not in self._CACHE:
            self._CACHE[key]=self._build_offline_policy(cfg)
        self.edges,self.xi,self.offline_price,self.offline_activation=self._CACHE[key]

    @classmethod
    def _fit_channel_model(cls,cfg):
        # Mission geometry (phase and patrol rate) is known offline, as assumed
        # by the statistical CMDP baseline.  Fading and Markov-state samples
        # use an independent stream, so the evaluated trace is never replayed.
        train=clone(cfg,T=cls.TRAIN_SLOTS)
        env=Env(train)
        env.rng=np.random.default_rng(RNG_MASTER+731000+cfg.seed)
        env.beta=env.rng.integers(0,2,train.N)
        env.iis=(env.rng.random(train.N)<train.p_ihigh).astype(int)
        env.ev=(env.rng.random(train.N)<train.p_event).astype(int)
        trace=[]
        zero={n:0.0 for n in range(train.N)}
        for _ in range(cls.TRAIN_SLOTS):
            obs=env.observe(); env._last=obs; trace.append(obs['pth'].copy())
            env.step([],zero)
        trace=np.asarray(trace)
        edges=np.zeros((train.N,cls.QBINS-1))
        labels=np.zeros_like(trace,dtype=int)
        omega=np.zeros((train.N,cls.QBINS+1))
        trans=np.zeros((train.N,cls.QBINS+1,cls.QBINS+1))
        for n in range(train.N):
            good=trace[:,n]<=train.Pmax
            vals=trace[good,n]
            if len(vals)<cls.QBINS:
                edges[n]=np.linspace(0.25,0.75,cls.QBINS-1)*train.Pmax
            else:
                edges[n]=np.quantile(vals,np.arange(1,cls.QBINS)/cls.QBINS)
            q=np.where(good,np.digitize(trace[:,n],edges[n]),cls.QBINS)
            labels[:,n]=q
            for j in range(cls.QBINS):
                v=trace[q==j,n]
                omega[n,j]=float(np.mean(v)) if len(v) else train.Pmax
            omega[n,cls.QBINS]=0.0
            counts=np.full((cls.QBINS+1,cls.QBINS+1),1e-3)
            np.add.at(counts,(q[:-1],q[1:]),1.0)
            trans[n]=counts/counts.sum(axis=1,keepdims=True)
        return edges,omega,trans

    @classmethod
    def _solve_one(cls,P,omega,pbar,price):
        X,Q=cls.XMAX,cls.QBINS+1; S=X*Q
        # z=[mu(0:S), y(0:S)], where y is active occupancy.
        cost=np.zeros(2*S)
        for x in range(X): cost[x*Q:(x+1)*Q]=x+1
        cost[S:]=price
        Aeq=[]; beq=[]
        for xp in range(X):
            for qp in range(Q):
                row=np.zeros(2*S); row[xp*Q+qp]=1.0
                for x in range(X):
                    xn=min(x+1,X-1)
                    for q in range(Q):
                        k=x*Q+q; prob=P[q,qp]
                        if xn==xp: row[k]-=prob
                        if xp==0: row[S+k]-=prob
                        if xn==xp: row[S+k]+=prob
                Aeq.append(row); beq.append(0.0)
        row=np.zeros(2*S); row[:S]=1.0
        Aeq.append(row); beq.append(1.0)
        Aub=[]; bub=[]
        # Active occupancy cannot exceed total occupancy.
        for k in range(S):
            row=np.zeros(2*S); row[S+k]=1.0; row[k]=-1.0
            Aub.append(row); bub.append(0.0)
        row=np.zeros(2*S)
        for x in range(X): row[S+x*Q:S+(x+1)*Q]=omega
        Aub.append(row); bub.append(pbar)
        bounds=[(0.0,1.0)]*(2*S)
        for x in range(X): bounds[S+x*Q+cls.QBINS]=(0.0,0.0)
        sol=linprog(cost,A_ub=np.asarray(Aub),b_ub=np.asarray(bub),
                    A_eq=np.asarray(Aeq),b_eq=np.asarray(beq),bounds=bounds,
                    method='highs')
        if not sol.success:
            raise RuntimeError("Tang-CMDP offline LP failed: "+sol.message)
        mu=sol.x[:S].reshape(X,Q); y=sol.x[S:].reshape(X,Q)
        return mu,y,float(y.sum())

    @classmethod
    def _build_offline_policy(cls,cfg):
        edges,omega,trans=cls._fit_channel_model(cfg)
        def solve_all(price):
            out=[cls._solve_one(trans[n],omega[n],cfg.pbar,price) for n in range(cfg.N)]
            return out,sum(v[2] for v in out)
        low,lo_act=solve_all(0.0)
        if lo_act<=cfg.M+1e-8:
            mixed=low; price=0.0; activation=lo_act
        else:
            wl,wh=0.0,1.0; high,hi_act=solve_all(wh)
            while hi_act>cfg.M and wh<1e6:
                wl=wh; low,lo_act=high,hi_act; wh*=2.0
                high,hi_act=solve_all(wh)
            for _ in range(9):
                wm=0.5*(wl+wh); mid,mid_act=solve_all(wm)
                if mid_act>cfg.M: wl,low,lo_act=wm,mid,mid_act
                else: wh,high,hi_act=wm,mid,mid_act
            nu=np.clip((cfg.M-hi_act)/(lo_act-hi_act+1e-12),0.0,1.0)
            mixed=[]
            for a,b in zip(low,high):
                mu=nu*a[0]+(1-nu)*b[0]; y=nu*a[1]+(1-nu)*b[1]
                mixed.append((mu,y,float(y.sum())))
            price=nu*wl+(1-nu)*wh; activation=sum(v[2] for v in mixed)
        xi=[]
        for mu,y,_ in mixed:
            ratio=np.divide(y,mu,out=np.zeros_like(y),where=mu>1e-8)
            ratio=np.clip(ratio,0.0,1.0)
            # Tang et al., Eq. (32): states outside the positive recurrent
            # support inherit the threshold action instead of becoming an
            # accidental never-transmit state after numerical LP truncation.
            for q in range(cls.QBINS):
                for x in range(cls.XMAX):
                    if x==cls.XMAX-1 or mu[x,q]<=1e-8 or (x>0 and ratio[x-1,q]>=1-1e-8):
                        ratio[x,q]=1.0
            ratio[:,cls.QBINS]=0.0
            xi.append(ratio)
        return edges,np.asarray(xi),float(price),float(activation)

    def decide(self,env,obs):
        q=np.empty(self.cfg.N,dtype=int)
        for n in range(self.cfg.N):
            q[n]=self.QBINS if not obs['deliverable'][n] else np.digitize(obs['pth'][n],self.edges[n])
        x=np.minimum(obs['A'].astype(int),self.XMAX)-1
        request=self.rng.random(self.cfg.N)<self.xi[np.arange(self.cfg.N),x,q]
        cand=np.flatnonzero(request & obs['deliverable'])
        if len(cand)>self.cfg.M:
            cand=self.rng.choice(cand,size=self.cfg.M,replace=False)
        power={n:0.0 for n in range(self.cfg.N)}
        for n in cand: power[int(n)]=float(obs['pth'][n])
        return [int(n) for n in cand],power

ANALYTIC = [RoundRobin, MaxAgeFirst, MaxWeight, WhittleAoI, Opportunistic,
            VoIGreedy, AoIIWhittle, ZhouLinCW, DPP_NoValue, CostWhittle,
            FixedPriceThr, AoIEnergyOnline, TangCMDP, FreshSkyWhittle, FreshSkyDPP]

# --------------------------------------------------------------------------
# DRL baseline: actor-critic (REINFORCE + value baseline) neural scheduler
#   learns a per-UAV score; selects top-M; power=p_th. Reward = -sum w*A - c_e*energy.
#   Represents learning-based schedulers (e.g., DRL RAN/UAV control) that lack a
#   feasibility guarantee -- a black box needing training.
# --------------------------------------------------------------------------
import torch, torch.nn as nn
torch.manual_seed(RNG_MASTER)
DRL_FEATS = 6

class DRLNet(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(DRL_FEATS,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        self.score=nn.Linear(h,1); self.val=nn.Linear(h,1)
    def forward(self,x):
        z=self.body(x); return self.score(z).squeeze(-1), self.val(z).squeeze(-1)

def drl_feats(env,obs):
    c=env.cfg
    A=obs['A']/50.0; w=obs['w']/c.wev; pth=np.minimum(obs['pth']/c.Pmax,2.0)
    Q=np.tanh(obs['Q']/50.0); dl=obs['deliverable'].astype(np.float32); be=obs['beta'].astype(np.float32)
    return np.stack([A,w,pth,Q,dl,be],axis=1).astype(np.float32)

def drl_select(score, deliverable, M, sample=True):
    avail=torch.tensor(deliverable.astype(bool)); chosen=[]; logp=0.0
    for _ in range(M):
        if int(avail.sum())==0: break
        logits=score.clone(); logits[~avail]=-1e9
        p=torch.softmax(logits,dim=0)
        i=(torch.multinomial(p,1).item() if sample else int(torch.argmax(p).item()))
        logp=logp+torch.log(p[i]+1e-12); chosen.append(i); avail[i]=False
    return chosen,logp

def train_drl(cfg, episodes=300, horizon=500, lr=3e-3, c_e=2.0, gamma=0.99):
    net=DRLNet(); opt=torch.optim.Adam(net.parameters(),lr=lr)
    for ep in range(episodes):
        env=Env(clone(cfg,seed=100+ep)); logps=[]; rews=[]; vals=[]
        for t in range(horizon):
            obs=env.observe(); env._last=obs
            x=torch.tensor(drl_feats(env,obs)); score,val=net(x)
            chosen,logp=drl_select(score,obs['deliverable'],cfg.M,sample=True)
            power={n:0.0 for n in range(cfg.N)}
            for n in chosen: power[n]=obs['pth'][n]
            s,p=env.step(chosen,power)
            r=-float(np.sum(obs['w']*env.A)) - c_e*float(np.sum(p))
            logps.append(logp); rews.append(r); vals.append(val.mean())
        R=0.0; returns=[]
        for r in reversed(rews): R=r+gamma*R; returns.insert(0,R)
        returns=torch.tensor(returns,dtype=torch.float32)
        returns=(returns-returns.mean())/(returns.std()+1e-6)
        valst=torch.stack(vals); adv=returns-valst.detach()
        loss=-(torch.stack(logps)*adv).mean() + 0.5*((valst-returns)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net

DRL_MODEL=None
class DRLPolicy(Policy):
    name="Deep Index Policy (RL)"; ref="learned neural index (policy-gradient Whittle-style, Zamir-Hou'24/26), no feasibility guarantee"
    def decide(self, env, obs):
        with torch.no_grad():
            score,_=DRL_MODEL(torch.tensor(drl_feats(env,obs)))
        chosen,_=drl_select(score,obs['deliverable'],self.cfg.M,sample=False)
        power={n:0.0 for n in range(self.cfg.N)}
        for n in chosen:
            if obs['pth'][n]<=self.cfg.Pmax: power[n]=obs['pth'][n]
        return [n for n in chosen if power[n]>0], power

# --------------------------------------------------------------------------
# Episode runner + metrics
# --------------------------------------------------------------------------
def run_episode(PolicyCls, cfg, warmup=200):
    cfg=cfg.derived()
    env=Env(cfg); pol=PolicyCls(cfg)
    N=cfg.N
    wAoI=0.0; cnt=0; powsum=np.zeros(N); Qsum=0.0
    peakA=[]; per_region_A=np.zeros(N)
    # event-detection latency: slots from event onset to next fresh delivery
    ev_prev=env.ev.copy(); ev_start=-np.ones(N); det_lat=[]
    t0=time.time()
    for t in range(cfg.T):
        obs=env.observe(); env._last=obs
        sched,power=pol.decide(env,obs)
        s,p=env.step(sched,power)
        if t>=warmup:
            wAoI += float(np.sum(obs['w']*env.A)); cnt+=1
            powsum+=p; Qsum+=float(np.sum(env.Q))
            peakA.append(float(np.max(env.A))); per_region_A+=env.A
        # event latency bookkeeping
        newev=env.ev
        for n in range(N):
            if ev_prev[n]==0 and newev[n]==1: ev_start[n]=t
            if ev_start[n]>=0 and s[n]==1:
                det_lat.append(t-ev_start[n]); ev_start[n]=-1
        ev_prev=newev.copy()
    dur=time.time()-t0
    avgpow=powsum/cnt
    per_region_A/=cnt
    jain=(per_region_A.sum()**2)/(N*np.sum(per_region_A**2)+1e-12)
    return dict(policy=pol.name, ref=pol.ref,
                wAoI=wAoI/cnt,
                peakAoI=float(np.mean(peakA)),
                p95AoI=float(np.percentile(peakA,95)),
                avg_power=float(np.mean(avgpow)),
                max_avg_power=float(np.max(avgpow)),
                budget=cfg.pbar,
                energy_ok=bool(np.all(avgpow<=cfg.pbar*1.02)),
                Qbacklog=Qsum/cnt,
                det_latency=float(np.mean(det_lat)) if det_lat else float('nan'),
                jain=float(jain),
                ms_per_slot=1e3*dur/cfg.T)

def run_multi(PolicyCls, cfg, seeds=(0,1,2,3,4)):
    rows=[run_episode(PolicyCls, clone(cfg, seed=s)) for s in seeds]
    df=pd.DataFrame(rows); agg={}
    for k in ['wAoI','peakAoI','p95AoI','avg_power','max_avg_power','Qbacklog',
              'det_latency','jain','ms_per_slot']:
        agg[k]=float(df[k].mean()); agg[k+'_std']=float(df[k].std())
    agg['policy']=rows[0]['policy']; agg['ref']=rows[0]['ref']
    agg['budget']=rows[0]['budget']; agg['energy_ok']=bool(df['energy_ok'].all())
    return agg

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG=os.path.join(os.path.dirname(__file__),"figs")
RES=os.path.join(os.path.dirname(__file__),"results")
os.makedirs(FIG,exist_ok=True); os.makedirs(RES,exist_ok=True)
VDEF=200.0
plt.rcParams.update({'font.size':11,'figure.dpi':150,'savefig.bbox':'tight'})
# grayscale-safe line styles: distinct (linestyle,marker) so figures read in B&W
STYLES=[('-','o','#1f77b4'),('--','s','#d62728'),(':','^','#2ca02c'),('-.','D','#9467bd')]

def short(name): return name.replace(" (ours)","*").replace("Round-Robin","RR")

def _hatch_infeasible(bars, feas):
    for b,ok in zip(bars, feas):
        if not ok: b.set_hatch('////')

def E1(base):
    print("\n== E1: default comparison ==")
    pols=ANALYTIC + ([DRLPolicy] if DRL_MODEL is not None else [])
    rows=[run_multi(P, clone(base,V=VDEF)) for P in pols]
    df=pd.DataFrame(rows).sort_values('wAoI').reset_index(drop=True)
    df.to_csv(os.path.join(RES,"summary_default.csv"),index=False)
    cols=['policy','wAoI','wAoI_std','peakAoI','det_latency','max_avg_power','energy_ok','Qbacklog','ref']
    print(df[cols].to_string(index=False))
    # LaTeX table
    with open(os.path.join(RES,"table_default.tex"),"w") as f:
        f.write("\\begin{tabular}{lrrrrc}\n\\hline\nPolicy & W-AoI & Peak & Det.\\ lat.\\ & Avg.\\ pow.\\ (mW) & Feas.\\\\\n\\hline\n")
        for _,r in df.iterrows():
            f.write(f"{r['policy']} & {r['wAoI']:.1f} & {r['peakAoI']:.1f} & {r['det_latency']:.2f} & {r['max_avg_power']:.2f} & {'yes' if r['energy_ok'] else 'NO'}\\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    # bar plot (feasible green, infeasible red)
    fig,ax=plt.subplots(figsize=(7,3.4))
    colors=['#2ca02c' if ok else '#d62728' for ok in df['energy_ok']]
    bars=ax.bar([short(p) for p in df['policy']], df['wAoI'], yerr=df['wAoI_std'],
           color=colors, edgecolor='k', linewidth=.6, capsize=3)
    _hatch_infeasible(bars, df['energy_ok'])
    ax.set_ylabel("Time-avg weighted AoI"); plt.setp(ax.get_xticklabels(),rotation=35,ha='right')
    ax.set_title("Weighted AoI (solid=feasible, hatched=budget-violating)")
    fig.savefig(os.path.join(FIG,"e1_wAoI_bar.png")); plt.close(fig)
    return df

def E2(base):
    print("\n== E2: energy feasibility ==")
    rows=[run_multi(P, clone(base,V=VDEF)) for P in ANALYTIC]
    df=pd.DataFrame(rows).sort_values('max_avg_power')
    fig,ax=plt.subplots(figsize=(7,3.2))
    colors=['#2ca02c' if ok else '#d62728' for ok in df['energy_ok']]
    bars=ax.bar([short(p) for p in df['policy']], df['max_avg_power'], color=colors, edgecolor='k', linewidth=.6)
    _hatch_infeasible(bars, df['energy_ok'])
    ax.axhline(base.pbar, ls='--', color='k', label=f"budget $\\bar p$={base.pbar:.0f} mW")
    ax.set_ylabel("Max per-UAV avg power (mW)"); ax.legend()
    plt.setp(ax.get_xticklabels(),rotation=35,ha='right')
    ax.set_title("Energy-agnostic baselines violate the budget")
    fig.savefig(os.path.join(FIG,"e2_energy.png")); plt.close(fig)

def E3_money(base):
    print("\n== E3: money plot (V sweep) ==")
    Vs=[1,2,5,10,20,50,100,200,500]
    aoi=[]; q=[]; pw=[]
    for Vv in Vs:
        r=run_multi(FreshSkyDPP, clone(base,V=Vv), seeds=(0,1,2,3))
        aoi.append(r['wAoI']); q.append(r['Qbacklog']); pw.append(r['max_avg_power'])
        print(f"  V={Vv:4d} wAoI={r['wAoI']:6.1f} Qbacklog={r['Qbacklog']:6.0f} avgpow={r['max_avg_power']:.2f}")
    fig,ax=plt.subplots(figsize=(5.6,3.6))
    ax.set_xscale('log'); l1=ax.plot(Vs,aoi,'o-',color='#1f77b4',label="weighted AoI")
    ax.set_xlabel("Lyapunov knob $V$"); ax.set_ylabel("weighted AoI",color='#1f77b4')
    ax.tick_params(axis='y',labelcolor='#1f77b4')
    ax2=ax.twinx(); l2=ax2.plot(Vs,q,'s--',color='#d62728',label="energy backlog $\\sum Q_n$")
    ax2.set_ylabel("energy backlog",color='#d62728'); ax2.tick_params(axis='y',labelcolor='#d62728')
    ax.set_title("AoI-energy tradeoff: $[O(1/V),\\,O(V)]$ (Thm 2)")
    ls=l1+l2; ax.legend(ls,[x.get_label() for x in ls],loc='upper center',fontsize=9)
    fig.savefig(os.path.join(FIG,"e3_moneyplot.png")); plt.close(fig)
    pd.DataFrame({'V':Vs,'wAoI':aoi,'Qbacklog':q,'avgpow':pw}).to_csv(os.path.join(RES,"moneyplot.csv"),index=False)

def E4_robust(base):
    print("\n== E4: robustness ==")
    key=[MaxWeight, DPP_NoValue, FreshSkyDPP]
    # (a) blockage burstiness: mean NLoS sojourn
    xs=[2,4,6,8,12,16]
    fig,ax=plt.subplots(figsize=(5.4,3.4))
    for i,P in enumerate(key):
        ys=[run_multi(P, clone(base,V=VDEF,mean_sojourn_N=x), seeds=(0,1,2))['wAoI'] for x in xs]
        ls,mk,cl=STYLES[i]; ax.plot(xs,ys,ls=ls,marker=mk,color=cl,label=short(P.name))
    ax.set_xlabel("mean NLoS dwell (slots) = blockage burstiness"); ax.set_ylabel("weighted AoI")
    ax.legend(fontsize=9); ax.set_title("Robustness to correlated blockage")
    fig.savefig(os.path.join(FIG,"e4_blockage.png")); plt.close(fig)
    # (b) interference
    xs=[0.1,0.2,0.3,0.4,0.5]
    fig,ax=plt.subplots(figsize=(5.4,3.4))
    for i,P in enumerate(key):
        ys=[run_multi(P, clone(base,V=VDEF,p_ihigh=x), seeds=(0,1,2))['wAoI'] for x in xs]
        ls,mk,cl=STYLES[i]; ax.plot(xs,ys,ls=ls,marker=mk,color=cl,label=short(P.name))
    ax.set_xlabel("interference probability $p_{I}$"); ax.set_ylabel("weighted AoI")
    ax.legend(fontsize=9); ax.set_title("Robustness to interference")
    fig.savefig(os.path.join(FIG,"e4_interference.png")); plt.close(fig)

def E5_scale(base):
    print("\n== E5: scale ==")
    key=[MaxWeight, DPP_NoValue, FreshSkyDPP]
    Ns=[6,9,12,16,20]
    fig,ax=plt.subplots(figsize=(5.4,3.4))
    for i,P in enumerate(key):
        ys=[run_multi(P, clone(base,V=VDEF,N=n,M=max(2,n//4)), seeds=(0,1,2))['wAoI'] for n in Ns]
        ls,mk,cl=STYLES[i]; ax.plot(Ns,ys,ls=ls,marker=mk,color=cl,label=short(P.name))
    ax.set_xlabel("number of UAVs $N$ (M=N/4)"); ax.set_ylabel("weighted AoI")
    ax.legend(fontsize=9); ax.set_title("Scalability")
    fig.savefig(os.path.join(FIG,"e5_scale.png")); plt.close(fig)

def E6_latency(base):
    print("\n== E6: event-detection latency ==")
    lst=[FreshSkyDPP, DPP_NoValue, MaxWeight, MaxAgeFirst, Opportunistic]
    if DRL_MODEL is not None: lst.append(DRLPolicy)
    rows=[run_multi(P, clone(base,V=VDEF)) for P in lst]
    df=pd.DataFrame(rows).sort_values('det_latency')
    fig,ax=plt.subplots(figsize=(5.2,3.2))
    colors=['#2ca02c' if ok else '#d62728' for ok in df['energy_ok']]
    bars=ax.bar([short(p) for p in df['policy']], df['det_latency'], color=colors, edgecolor='k',linewidth=.6)
    _hatch_infeasible(bars, df['energy_ok'])
    ax.set_ylabel("event-detection latency (slots)")
    plt.setp(ax.get_xticklabels(),rotation=30,ha='right')
    ax.set_title("Time from event onset to fresh delivery")
    fig.savefig(os.path.join(FIG,"e6_latency.png")); plt.close(fig)

if __name__=="__main__":
    t0=time.time()
    base=Cfg().derived()
    print("== calibration ==")
    print(f"gamma={base.gamma:.3f} ({10*np.log10(base.gamma):.1f} dB), "
          f"sigma2={10*np.log10(base.sigma2):.1f} dBm, pbar={base.pbar:.1f} mW, Pmax={base.Pmax:.0f} mW, V={VDEF:.0f}")
    e=Env(base); dl=[]
    for _ in range(500):
        o=e.observe(); e._last=o; dl.append(o['deliverable'].mean())
        e.step([],{n:0.0 for n in range(base.N)})
    print(f"avg deliverable fraction = {np.mean(dl):.2f}")
    print("== training DRL baseline (this takes ~1-3 min) ==")
    tD=time.time(); DRL_MODEL=train_drl(base)
    rD=run_multi(DRLPolicy, clone(base,V=VDEF), seeds=(0,1,2))
    print(f"  DRL trained in {time.time()-tD:.0f}s -> wAoI={rD['wAoI']:.1f} "
          f"feasible={rD['energy_ok']} avgpow={rD['max_avg_power']:.2f}mW")
    df=E1(base); E2(base); E3_money(base); E4_robust(base); E5_scale(base); E6_latency(base)
    print(f"\nAll experiments done in {time.time()-t0:.0f}s. Figures -> {FIG}, results -> {RES}")
