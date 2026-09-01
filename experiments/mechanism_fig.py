# -*- coding: utf-8 -*-
"""Overall FreshSky mechanism diagram: per-slot closed loop.
Short labels inside boxes (formula/legend go in the caption) so nothing
overflows."""
import os, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
FIG=os.path.join(os.path.dirname(__file__),"figs")

def box(ax,cx,cy,w,h,text,fc):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,
        boxstyle="round,pad=0.02,rounding_size=0.08",fc=fc,ec='k',lw=1.2))
    ax.text(cx,cy,text,ha='center',va='center',fontsize=9.5)

def arr(ax,x0,x1,y,color='k',ls='-'):
    ax.add_patch(FancyArrowPatch((x0,y),(x1,y),arrowstyle='-|>',mutation_scale=14,lw=1.4,color=color,ls=ls))

fig,ax=plt.subplots(figsize=(7.2,2.1)); ax.set_xlim(0,12.6); ax.set_ylim(0,3.4); ax.axis('off')
cy=2.15; w=2.7; h=1.05
cx=[1.55,4.55,7.35,10.15]
labels=["Per-UAV state\n$(A_n,\\,w_n,\\,\\theta_n,\\,Q_n)$",
        "Belief-Whittle\nindex $\\mathcal{B}_n$",
        "Top-$M$\nselection",
        "Transmit\n$+$ ACK/NACK"]
cols=["#dbe9f6","#e7f0da","#f6e6cf","#f6d9d9"]
for c,t,fc in zip(cx,labels,cols): box(ax,c,cy,w,h,t,fc)
for i in range(3): arr(ax,cx[i]+w/2,cx[i+1]-w/2,cy)
# feedback loop underneath
yb=0.72
ax.add_patch(FancyArrowPatch((cx[3],cy-h/2),(cx[3],yb),arrowstyle='-',lw=1.4,color='#c44e52'))
ax.add_patch(FancyArrowPatch((cx[3],yb),(cx[0],yb),arrowstyle='-',lw=1.4,color='#c44e52',ls='--'))
ax.add_patch(FancyArrowPatch((cx[0],yb),(cx[0],cy-h/2),arrowstyle='-|>',mutation_scale=14,lw=1.4,color='#c44e52',ls='--'))
ax.text((cx[0]+cx[3])/2,yb-0.34,"feedback: update belief $\\theta_n$, energy queue $Q_n$, age $A_n$",
        ha='center',va='center',fontsize=8.5,color='#c44e52')
fig.savefig(os.path.join(FIG,"fig_mechanism.png"),dpi=160,bbox_inches='tight'); plt.close(fig)
print("saved fig_mechanism.png")
