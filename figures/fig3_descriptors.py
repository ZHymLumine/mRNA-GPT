"""De novo sequence descriptors, three arms against real genes -- panels e-h
of Figure 3.

This is the unconstrained arm: the model writes a coding sequence with no
target protein, so what it emits reflects its own learned codon preferences
rather than choices forced by a fixed protein. That makes it the cleaner test
of what fine-tuning changed.

Each panel plots the three arms (pretrained / fine-tuned on high-property genes
/ fine-tuned on low-property genes) against two reference lines: the mean of
real top-quartile and real bottom-quartile genes for that task. The question is
not "which arm is highest" -- none of these four descriptors has a universal
direction -- but whether the high arm sits on the real high line and the low
arm on the real low side of it.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import C, panel_label, save                          # noqa: E402
from sft.paths import RUNS                                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = str(RUNS)
REAL = json.load(open(os.path.join(HERE, "real_gene_metrics.json")))

# (task key, column title, high-file, low-file, arm keys in those files)
TASKS = [("expression", "Fungal transcript expression",
          f"{R}/fungal_sft/generation/evaluator_comparison.json",
          f"{R}/fungal_sft/generation/evaluator_comparison_low.json",
          ("pretrained_eukaryote", "fungal_sft", "fungal_sft_low")),
         ("stability", "mRNA stability",
          f"{R}/stability_sft/generation/evaluator_comparison.json",
          f"{R}/stability_sft/generation/evaluator_comparison_low.json",
          ("pretrained_archaea", "stability_sft", "stability_sft_low")),
         # Bacterial arms come from the swept re-runs, whose cosine schedule
         # completes; runs/bacexp_sft's control moved toward the high-expression
         # profile and is superseded. The pretrained arm is unchanged, so it is
         # read from whichever of the two files carries it.
         ("bacexp", "Bacteria protein expression",
          f"{R}/bacexp_sweep/high_lr1e5/evaluator_comparison.json",
          f"{R}/bacexp_sweep/low_lr1e5/evaluator_comparison.json",
          ("mrna_gpt_pretrained", "mrna_gpt_sft", "mrna_gpt_sft_LOW"))]

METRICS = [("cai", "CAI"), ("tai", "tAI"), ("gc3", "GC3"), ("mfe_per_nt", "MFE per nt")]
ARMS = [("Pretrained", C["pretrained"]), ("Fine-tuned (high)", C["sft"]),
        ("Fine-tuned (low, control)", C["sft_low"])]


def arms_for(hi_path, lo_path, keys):
    """-> [{metric: value} or None] for pretrained / high / low."""
    pre, high, low = None, None, None
    if os.path.exists(hi_path):
        d = json.load(open(hi_path))
        if keys[0] in d:
            pre = {m: d[keys[0]][m]["mean"] for m, _ in METRICS}
        if keys[1] in d:
            high = {m: d[keys[1]][m]["mean"] for m, _ in METRICS}
    if os.path.exists(lo_path):
        d = json.load(open(lo_path))
        if keys[2] in d:
            low = {m: d[keys[2]][m]["mean"] for m, _ in METRICS}
    return [pre, high, low]


def main():
    cols = [(k, t, arms_for(h, l, a)) for k, t, h, l, a in TASKS]
    cols = [(k, t, a) for k, t, a in cols if any(x for x in a)]

    fig, axes = plt.subplots(len(METRICS), len(cols),
                             figsize=(2.35 * len(cols) + 1.4, 6.0), squeeze=False)
    for ci, (key, title, arms) in enumerate(cols):
        hi, lo = REAL[key]["real_high"], REAL[key]["real_low"]
        for ri, (mk, mlabel) in enumerate(METRICS):
            ax = axes[ri][ci]
            band = sorted((hi[mk], lo[mk]))
            ax.axvspan(band[0], band[1], color=C["real_high"], alpha=0.18, zorder=0)
            ax.axvline(hi[mk], color=C["real_high"], ls="--", lw=1.0, zorder=2)
            ax.axvline(lo[mk], color=C["real_low"], ls=":", lw=1.0, zorder=2)
            for ai, (alabel, ac) in enumerate(ARMS):
                if arms[ai] is None:
                    continue
                v = arms[ai][mk]
                ax.plot([v, v], [ai - 0.30, ai + 0.30], color=ac, lw=3.2,
                        solid_capstyle="butt", zorder=4)
                ax.plot(v, ai, "o", color=ac, ms=4.2, zorder=5,
                        markeredgecolor="white", markeredgewidth=0.6)
            ax.set_yticks(range(len(ARMS)))
            ax.set_yticklabels([a for a, _ in ARMS] if ci == 0 else [], fontsize=6)
            ax.set_ylim(len(ARMS) - 0.45, -0.55)
            ax.tick_params(axis="x", labelsize=6)
            ax.grid(axis="x", lw=0.3, color="0.93")
            ax.set_axisbelow(True)
            # widen slightly so markers at the edge are not clipped
            vals = [a[mk] for a in arms if a] + [hi[mk], lo[mk]]
            pad = 0.10 * (max(vals) - min(vals) + 1e-9)
            ax.set_xlim(min(vals) - pad, max(vals) + pad)
            if ci == 0:
                ax.set_ylabel(mlabel, fontsize=7.5)
            if ri == 0:
                ax.set_title(title, fontsize=7.5, pad=4)
            # the real high-vs-low gap of every panel is in the source data
            # table; printing it here collided with the reference lines

    h = [plt.Line2D([], [], marker="o", ls="", color=C["pretrained"],
                    label="Pretrained"),
         plt.Line2D([], [], marker="o", ls="", color=C["sft"],
                    label="Fine-tuned (high)"),
         plt.Line2D([], [], marker="o", ls="", color=C["sft_low"],
                    label="Control (low)"),
         plt.Line2D([], [], color=C["real_high"], ls="--", lw=1.2,
                    label="Real top-quartile genes"),
         plt.Line2D([], [], color=C["real_low"], ls=":", lw=1.2,
                    label="Real bottom-quartile genes"),
         plt.Rectangle((0, 0), 1, 1, facecolor=C["real_high"], alpha=0.18,
                       edgecolor="0.6", linewidth=0.3,
                       label="Range spanned by real genes")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=6.2, frameon=False,
               bbox_to_anchor=(0.5, -0.012))
    for i, lab in enumerate("abcd"):
        panel_label(axes[i][0], lab, dx=-0.62, dy=1.10)
    fig.tight_layout(rect=(0, 0.065, 1, 1), h_pad=1.2, w_pad=0.9)
    save(fig, os.path.join(HERE, "figure3_descriptors"))

    tsv = os.path.join(HERE, "figure3_descriptors.tsv")
    with open(tsv, "w") as fh:
        fh.write("task\tarm\t" + "\t".join(m for m, _ in METRICS) + "\n")
        for key, title, arms in cols:
            for (alabel, _), a in zip(ARMS, arms):
                if a:
                    fh.write(f"{key}\t{alabel}\t" +
                             "\t".join(f"{a[m]:.4f}" for m, _ in METRICS) + "\n")
            for ref in ("real_high", "real_low"):
                fh.write(f"{key}\t[{ref} genes]\t" +
                         "\t".join(f"{REAL[key][ref][m]:.4f}" for m, _ in METRICS) + "\n")
    print("wrote " + tsv)
    for key, title, arms in cols:
        miss = [a for (a, _), v in zip(ARMS, arms) if v is None]
        if miss:
            print(f"  pending for {title}: {', '.join(miss)}")
    for key, t, h_, l_, a_ in TASKS:
        if key not in [c[0] for c in cols]:
            print(f"  pending (no data yet): {t}")


if __name__ == "__main__":
    main()
