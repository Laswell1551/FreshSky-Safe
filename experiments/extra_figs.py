# -*- coding: utf-8 -*-
"""Extra figures for the FreshSky paper: system schematic + energy-queue trajectory."""
import os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patches as mp
import freshsky_sim as F
FIG=os.path.join(os.path.dirname(__file__),"figs")

def system_fig():
    fig,ax=plt.subplots(figsize=(6.6,3.3)); ax.set_xlim(0,10); ax.set_ylim(-0.6,5.6); ax.axis('off')
    ax.add_patch(mp.Rectangle((0,0),10,0.4,color='#d9d9d9'))
    ax.add_patch(mp.Rectangle((0.6,0.4),0.55,1.1,color='#4c72b0'))
    ax.plot([0.87,0.87],[1.5,2.05],color='k'); ax.plot(0.87,2.1,marker='^',color='k',ms=8)
    ax.text(0.87,-0.15,"GS / edge\n(scheduler)",ha='center',va='top',fontsize=8.5)
    ax.add_patch(mp.Rectangle((4.7,0.4),1.0,2.3,color='#8c8c8c'))
    ax.text(5.2,1.5,"blockage",ha='center',fontsize=8,color='white',rotation=90)
    uavs=[(2.5,3.8),(3.9,4.7),(6.3,4.3),(8.3,3.5)]
    for i,(x,y) in enumerate(uavs):
        ax.plot(x,y,marker='X',ms=13,color='#c44e52')
        ax.text(x,y+0.28,f"UAV {i+1}",ha='center',fontsize=8)
        ax.add_patch(mp.Arc((x,y-0.05),1.5,0.5,angle=0,theta1=200,theta2=340,color='#c44e52',ls=':',lw=1))
    for (x,y) in [(2.5,3.8),(3.9,4.7),(8.3,3.5)]:
        ax.annotate("",xy=(0.87,2.1),xytext=(x,y),arrowprops=dict(arrowstyle='<->',color='#55a868',lw=1.3))
    ax.annotate("",xy=(0.87,2.1),xytext=(6.3,4.3),arrowprops=dict(arrowstyle='<->',color='#c44e52',ls='--',lw=1.4))
    ax.plot(8.9,1.3,marker='*',ms=16,color='#dd8452'); ax.text(8.9,0.85,"interferer",ha='center',fontsize=8)
    for (x,y) in [(6.3,4.3),(8.3,3.5)]:
        ax.annotate("",xy=(x,y),xytext=(8.9,1.45),arrowprops=dict(arrowstyle='->',color='#dd8452',ls=':',lw=1))
    ax.text(2.0,2.5,"LoS uplink",color='#3d8a4f',fontsize=8)
    ax.text(4.9,3.5,"NLoS\n(blocked)",color='#c44e52',fontsize=8,ha='center')
    ax.text(5.0,5.35,r"$N$ UAVs share $M{<}N$ uplink sub-bands; scheduler picks who/where/what power each slot",
            ha='center',fontsize=8.5)
    fig.savefig(os.path.join(FIG,"fig_system.png"),dpi=150,bbox_inches='tight'); plt.close(fig)

def queue_fig():
    base=F.Cfg().derived()
    def traj(PolicyCls, V=50.0, T=1500):
        cfg=F.clone(base,V=V,T=T); env=F.Env(cfg); pol=PolicyCls(cfg); qs=[]
        for t in range(T):
            obs=env.observe(); env._last=obs
            sched,power=pol.decide(env,obs); env.step(sched,power); qs.append(float(env.Q.sum()))
        return np.array(qs)
    qf=traj(F.FreshSkyDPP); qm=traj(F.MaxWeight)
    fig,ax=plt.subplots(figsize=(5.4,3.2))
    ax.plot(qm,color='#c44e52',lw=1.6,ls='--',label="Max-Weight (energy-agnostic): diverges")
    ax.plot(qf,color='#55a868',lw=1.8,ls='-',label="FreshSky (ours): bounded")
    ax.set_xlabel("slot $t$"); ax.set_ylabel(r"energy backlog $\sum_n Q_n(t)$")
    ax.legend(fontsize=9,loc='upper left'); ax.set_title("Energy feasibility (Thm 1): FreshSky keeps queues bounded")
    fig.savefig(os.path.join(FIG,"e7_queue.png"),dpi=150,bbox_inches='tight'); plt.close(fig)
    print(f"final backlog: FreshSky={qf[-1]:.0f}  MaxWeight={qm[-1]:.0f}")

if __name__=="__main__":
    system_fig(); queue_fig(); print("extra figs done ->", FIG)
