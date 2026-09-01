# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) decision-time information schematic -> cross-type inherit -> param inherit
# (b) separation-architecture schematic -> cross-type inherit -> param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.

# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent / "freshsky_tmc_figures"
OUT.mkdir(exist_ok=True)


def box(ax, xy, width, height, text, face, edge=BLACK, fontsize=7,
        lw=0.7, radius=0.025, text_color=BLACK):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color=text_color, linespacing=1.15)
    return patch


def arrow(ax, start, end, color=BLACK, style="-|>", lw=0.8,
          connectionstyle="arc3"):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=8,
        linewidth=lw, color=color, connectionstyle=connectionstyle
    )
    ax.add_patch(patch)
    return patch


def make_information_timeline():
    fig, ax = plt.subplots(figsize=(89 / 25.4, 57 / 25.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.94, "Decision-time information at slot $t$",
            fontsize=8, fontweight="bold", va="top")

    y = 0.52
    ax.plot([0.08, 0.92], [y, y], color=BLACK, linewidth=1.0)
    points = [(0.16, "$t-d-1$"), (0.38, "$t-d$"),
              (0.62, "$t-1$"), (0.86, "$t$")]
    for x, label in points:
        ax.plot([x, x], [y - 0.025, y + 0.025], color=BLACK, linewidth=0.8)
        ax.text(x, y - 0.075, label, ha="center", va="top", fontsize=7)

    box(ax, (0.07, 0.62), 0.25, 0.18,
        "Confirmed history\nthrough $t-d-1$",
        "#DCEAF7", edge=CATEGORICAL[0])
    box(ax, (0.35, 0.62), 0.38, 0.18,
        "$d$ pending slots\nreceiver AoI latent\nlink state latent",
        "#F4E2E0", edge=CATEGORICAL[1], fontsize=6.2)
    box(ax, (0.76, 0.62), 0.19, 0.18,
        "Current\ndecision",
        "#E2F0E5", edge=CATEGORICAL[2])

    ax.annotate("", xy=(0.72, 0.59), xytext=(0.36, 0.59),
                arrowprops=dict(arrowstyle="<->", color=CATEGORICAL[1],
                                linewidth=0.8))
    ax.text(0.54, 0.565, "$d$ pending-feedback slots", ha="center", va="top",
            fontsize=6.5, color=CATEGORICAL[1])

    box(ax, (0.07, 0.16), 0.88, 0.20,
        "Known now: geometry, value $w_n(t)$, stored actions,\n"
        "and individual debt $Q_n(t)$\n"
        "Estimated now: posterior $\\rho_{n,k}(t)$, success $q_n(t)$,\n"
        "and receiver age $\\hat A_n(t)$",
        "#F5F5F5", edge="#666666", fontsize=5.8, radius=0.018)

    arrow(ax, (0.86, 0.50), (0.86, 0.38), color=CATEGORICAL[2])
    ax.text(0.96, 0.46, "schedule", rotation=90, ha="center", va="center",
            fontsize=6.5, color=CATEGORICAL[2])

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    save_cns_figure(fig, OUT / "fig_system")
    plt.close(fig)


def make_separation_architecture():
    fig, ax = plt.subplots(figsize=(89 / 25.4, 90 / 25.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.97, "FreshSky-Safe separation architecture",
            fontsize=8, fontweight="bold", va="top")
    ax.text(0.04, 0.915, "Statistical ranking (performance)",
            fontsize=7, fontweight="bold", color=CATEGORICAL[0])

    box(ax, (0.06, 0.77), 0.40, 0.10,
        "Delayed ACK/NACK\nBayesian model update",
        "#DCEAF7", edge=CATEGORICAL[0], fontsize=6.5)
    box(ax, (0.54, 0.77), 0.40, 0.10,
        "Pending history\nexpected receiver AoI",
        "#DCEAF7", edge=CATEGORICAL[0], fontsize=6.5)
    arrow(ax, (0.46, 0.82), (0.54, 0.82), color=CATEGORICAL[0])

    box(ax, (0.20, 0.61), 0.60, 0.10,
        "Myopic freshness priority $R_n(t)$\n"
        "(posterior benefit $-$ individual debt price)",
        "#DCEAF7", edge=CATEGORICAL[0], fontsize=6.5)
    arrow(ax, (0.26, 0.77), (0.39, 0.71), color=CATEGORICAL[0])
    arrow(ax, (0.74, 0.77), (0.61, 0.71), color=CATEGORICAL[0])

    box(ax, (0.30, 0.47), 0.40, 0.08,
        "Top-$M$ proposal $\\mathcal{S}_0(t)$",
        "#DCEAF7", edge=CATEGORICAL[0], fontsize=6.5)
    arrow(ax, (0.50, 0.61), (0.50, 0.55), color=CATEGORICAL[0])

    ax.plot([0.05, 0.95], [0.42, 0.42], color="#777777",
            linewidth=0.7, linestyle="--")
    ax.text(0.04, 0.39, "Exact accounting (safety)",
            fontsize=7, fontweight="bold", color=CATEGORICAL[1], va="top")

    box(ax, (0.16, 0.25), 0.68, 0.10,
        "Per-UAV queue-cap shield\n"
        "$[Q_n(t)-\\bar p_n]^++p_n^{\\mathrm{th}}(t)"
        "\\leq Q_{\\max}$",
        "#F4E2E0", edge=CATEGORICAL[1], fontsize=6.5)
    arrow(ax, (0.50, 0.47), (0.50, 0.35), color=CATEGORICAL[1])

    box(ax, (0.06, 0.08), 0.40, 0.10,
        "Admit\ntransmit at exact charged power",
        "#E2F0E5", edge=CATEGORICAL[2], fontsize=6.5)
    box(ax, (0.54, 0.08), 0.40, 0.10,
        "Reject\nzero controlled uplink power",
        "#F2F2F2", edge="#666666", fontsize=6.5)
    arrow(ax, (0.40, 0.25), (0.27, 0.18), color=CATEGORICAL[2])
    arrow(ax, (0.60, 0.25), (0.73, 0.18), color="#666666")

    arrow(ax, (0.27, 0.08), (0.12, 0.78), color="#666666",
          style="-|>", lw=0.7, connectionstyle="arc3,rad=-0.30")
    ax.text(0.015, 0.43, "feedback usable after $d$ pending slots",
            fontsize=6.3, rotation=90, va="center", color="#555555")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save_cns_figure(fig, OUT / "fig_mechanism")
    plt.close(fig)


if __name__ == "__main__":
    make_information_timeline()
    make_separation_architecture()
