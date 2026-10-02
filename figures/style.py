"""Shared figure style: Nature/Science-style results panels.

Conventions that make the five figures read as one set:
  * one sans-serif family, 7 pt base, panel letters 9 pt bold upper-left
  * no top/right spines, hairline axes, ticks pointing out
  * a fixed semantic palette -- the same arm is the same colour in every figure
  * real measured genes are never bars: they are reference lines/bands, because
    they are the scale the bars are read against, not another method
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,      # editable text in Illustrator
    "ps.fonttype": 42,
})

# semantic palette -- same meaning, same colour, in every figure
C = {
    "pretrained": "#9AA0A6",     # grey: the un-fine-tuned starting point
    "sft": "#1F6FB4",            # blue: property fine-tuned (the method)
    "sft_low": "#D1495B",        # red: negative control (opposite quartile)
    "real_high": "#2E7D32",      # green: real top-quartile genes
    "real_low": "#A5D6A7",       # pale green: real bottom-quartile genes
    "real_range": "#B0BEC5",     # neutral blue-grey: the band between the two
                                 # real sets.  Deliberately not green: the band
                                 # and the real-low line are different things and
                                 # two shades of one hue made them look alike.
    "cai_sample": "#7B68A6",     # purple: frequency-matching baseline
    "cai_max": "#E08214",        # orange: CAI maximisation
    "ld": "#00868B",             # teal: LinearDesign family
    "other": "#B0B0B0",          # other published baselines
    "archaea": "#4C72B0",
    "bacteria": "#DD8452",
    "eukaryote": "#55A868",
}

DOMAIN_C = {"archaea": C["archaea"], "bacteria": C["bacteria"], "eukaryote": C["eukaryote"]}


def panel_label(ax, letter: str, dx: float = -0.20, dy: float = 1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")


def hline(ax, y, color, label=None, ls="--", lw=0.9, zorder=0):
    ax.axhline(y, color=color, ls=ls, lw=lw, zorder=zorder, label=label)


def bars(ax, labels, values, colors, err=None, width=0.68, edge="none"):
    x = range(len(labels))
    b = ax.bar(x, values, width=width, color=colors, edgecolor=edge, linewidth=0.4,
               yerr=err, error_kw=dict(lw=0.6, capsize=1.8, capthick=0.6))
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    return b


def save(fig, path_stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}")
    plt.close(fig)
    print(f"wrote {path_stem}.pdf / .png")
