"""Figure 3. Property-specific fine-tuning enables de novo generation of mRNA
sequences with desired properties.

Two properties, two pretrained backbones: mRNA stability (mRNA-GPT-archaea) and
expression (mRNA-GPT-eukaryote). Each has three arms -- the pretrained model, the
model fine-tuned on the measured top quartile, and the negative control
fine-tuned on the measured bottom quartile with identical settings.

Everything is plotted in units of the real high/low gap for that property, so the
two properties share an axis and a bar can be read directly against what real
genes do: 0 = real bottom-quartile genes, 1 = real top-quartile genes.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import C, bars, panel_label, save                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "figure_data.json")))

PROPS = [("stability", "mRNA stability\n(mRNA-GPT-archaea)",
          ["pretrained_archaea", "stability_sft", "stability_sft_LOW"]),
         ("expression", "Expression\n(mRNA-GPT-eukaryote)",
          ["uncon_pretrained", "uncon_sft_HIGH", "uncon_sft_LOW"])]
ARM_C = [C["pretrained"], C["sft"], C["sft_low"]]
ARM_N = ["Pretrained", "Fine-tuned\n(top quartile)", "Control\n(bottom quartile)"]


def norm(v, lo, hi):
    return (v - lo) / (hi - lo)


def grouped(ax, values, ylabel, ref_lines=None, log=False, outside=None):
    """values[prop][arm]; two property groups of three bars.

    `outside[prop][arm]` marks arms whose codon profile lies outside the range
    spanned by real genes. Those bars are hatched: a likelihood-ratio or a
    predictor score computed there is an extrapolation, not a ranking.
    """
    n_arm = 3
    width = 0.26
    xs = np.arange(len(PROPS))
    for i in range(n_arm):
        h = [("///" if outside and outside[p][i] else "") for p in range(len(PROPS))]
        b = ax.bar(xs + (i - 1) * width, [values[p][i] for p in range(len(PROPS))],
                   width=width * 0.92, color=ARM_C[i], linewidth=0, label=ARM_N[i], zorder=3)
        for rect, hh in zip(b, h):
            if hh:
                rect.set_hatch(hh)
                rect.set_edgecolor("white")
                rect.set_linewidth(0.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([p[1] for p in PROPS])
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    for y, col, lab in (ref_lines or []):
        ax.axhline(y, color=col, ls="--", lw=0.9, zorder=1)
        ax.text(1.01, y, lab, transform=ax.get_yaxis_transform(), color=col,
                fontsize=6, va="center", ha="left")
    ax.grid(axis="y", lw=0.35, color="0.9", zorder=0)
    ax.set_axisbelow(True)


def main():
    dn, pr = D["denovo"], D["denovo_predictor"]
    js, llr, lgbm, box = [], [], [], []
    for key, _title, arms in PROPS:
        blk = dn[key]
        gap = blk["real_low"]["js_to_high"]              # real high -> real low distance
        js.append([blk["arms"][a]["js_to_high"] / gap for a in arms])
        lo, hi = blk["real_low"]["llr_high_low"], blk["real_high"]["llr_high_low"]
        llr.append([norm(blk["arms"][a]["llr_high_low"], lo, hi) for a in arms])
        p = pr[key]
        lgbm.append([norm(p[a]["mean"], p["real_low"], p["real_high"])
                     for a in ("pretrained", "sft", "sft_low")])
        box.append([[norm(v, p["real_low"], p["real_high"]) for v in p[a]["values"]]
                    for a in ("pretrained", "sft", "sft_low")])

    fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.0))
    outside = [[v > 1.0 for v in row] for row in js]         # beyond the real high-low span
    ax = axes[0, 0]
    grouped(ax, js, "Distance to real high-property genes\n(JS, units of real high–low gap)",
            ref_lines=[(1.0, C["real_low"], "real low"), ],  log=True)
    ax.axhspan(1e-3, 1.0, color=C["real_high"], alpha=0.07, zorder=0)
    ax.set_ylim(1e-3, 200)
    ax.text(0.02, 0.96, "lower = more like a real\ntop-quartile gene", transform=ax.transAxes,
            fontsize=6, va="top", color="0.35")
    ax.legend(loc="upper right", ncol=1, handlelength=1.1, borderpad=0.2)
    panel_label(ax, "a")

    ax = axes[0, 1]
    grouped(ax, llr, "Codon log-likelihood ratio\n(0 = real low, 1 = real high genes)",
            ref_lines=[(1.0, C["real_high"], "real high"), (0.0, C["real_low"], "real low")],
            outside=outside)
    ax.text(0.02, 0.97, "hatched: codon profile outside\nthe range of real genes (a)",
            transform=ax.transAxes, fontsize=6, va="top", color="0.35")
    panel_label(ax, "b")

    ax = axes[1, 0]
    grouped(ax, lgbm, "Predicted property, independent model\n(0 = real low, 1 = real high genes)",
            ref_lines=[(1.0, C["real_high"], "real high"), (0.0, C["real_low"], "real low")],
            outside=outside)
    panel_label(ax, "c")

    ax = axes[1, 1]
    pos, xt, xl = [], [], []
    for gi in range(len(PROPS)):
        for ai in range(3):
            p = gi * 3.6 + ai
            pos.append(p)
            bp = ax.boxplot([box[gi][ai]], positions=[p], widths=0.62, showfliers=False,
                            patch_artist=True, medianprops=dict(color="white", lw=0.9),
                            boxprops=dict(linewidth=0.4), whiskerprops=dict(linewidth=0.5),
                            capprops=dict(linewidth=0.5))
            bp["boxes"][0].set_facecolor(ARM_C[ai])
            bp["boxes"][0].set_edgecolor("0.3")
        xt.append(gi * 3.6 + 1)
        xl.append(PROPS[gi][1])
    ax.axhline(1.0, color=C["real_high"], ls="--", lw=0.9)
    ax.axhline(0.0, color=C["real_low"], ls="--", lw=0.9)
    ax.text(1.01, 1.0, "real high", transform=ax.get_yaxis_transform(),
            color=C["real_high"], fontsize=6, va="center")
    ax.text(1.01, 0.0, "real low", transform=ax.get_yaxis_transform(),
            color=C["real_low"], fontsize=6, va="center")
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("Predicted property per sequence\n(n = 500 per arm, normalised)")
    ax.grid(axis="y", lw=0.35, color="0.9"); ax.set_axisbelow(True)
    panel_label(ax, "d")

    fig.tight_layout(w_pad=2.4, h_pad=2.0)
    save(fig, os.path.join(HERE, "figure3_denovo"))


if __name__ == "__main__":
    main()
