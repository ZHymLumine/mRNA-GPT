"""Figure 2. Pretrained mRNA-GPT captures coding patterns across the three
domains of life.

Three models, one per domain, each pretrained on a homology-aware split of its
own corpus. The panels test whether the models learned domain-specific coding
patterns rather than a generic codon prior.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import C, DOMAIN_C, panel_label, save                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "figure_data.json")))
DOMS = ["archaea", "bacteria", "eukaryote"]
DLAB = ["Archaea", "Bacteria", "Eukaryote"]


def main():
    fig = plt.figure(figsize=(7.09, 5.1))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.42)

    # ---- a: cross-domain perplexity ----
    ax = fig.add_subplot(gs[0, 0])
    M = np.array([[D["cross_domain_ppl"][m][v] for v in DOMS] for m in DOMS])
    im = ax.imshow(M, cmap="viridis_r", aspect="auto")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > M.min() + 0.55 * (M.max() - M.min()) else "black",
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(3)); ax.set_xticklabels(DLAB, rotation=20, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(DLAB)
    ax.set_xlabel("Validation set"); ax.set_ylabel("Pretrained model")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Perplexity (PAD excluded)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("Each model is best on its own domain", fontsize=7, pad=4)
    panel_label(ax, "a")

    # ---- b: homology leakage ----
    ax = fig.add_subplot(gs[0, 1])
    thr = ["\u226550%", "\u226570%", "\u226590%", "\u226599%"]
    x = np.arange(len(thr)); w = 0.13
    for k, d in enumerate(DOMS):
        xr = x + (k - 1) * 2 * w - w / 2
        xo = x + (k - 1) * 2 * w + w / 2
        ax.bar(xr, D["leakage"][d]["random"], width=w * 0.9,
               color=DOMAIN_C[d], alpha=0.40, linewidth=0)
        ax.bar(xo, D["leakage"][d]["ours"], width=w * 0.9, color=DOMAIN_C[d], linewidth=0)
    # the homology-aware bars are at or near zero; print them so the panel is readable
    for k, d in enumerate(DOMS):
        for i, v in enumerate(D["leakage"][d]["ours"]):
            ax.text(x[i] + (k - 1) * 2 * w + w / 2, 2.0, f"{v:.2f}", rotation=90,
                    ha="center", va="bottom", fontsize=5.2, color=DOMAIN_C[d])
    ax.set_xticks(x); ax.set_xticklabels(thr)
    ax.set_xlabel("Nucleotide identity to a training sequence")
    ax.set_ylabel("Validation sequences with a\nhomolog in training (%)")
    ax.set_ylim(0, 100)
    h = ([plt.Rectangle((0, 0), 1, 1, color=DOMAIN_C[d], alpha=0.40) for d in DOMS]
         + [plt.Rectangle((0, 0), 1, 1, color=DOMAIN_C[d]) for d in DOMS])
    ax.legend(h, [f"{l} (random)" for l in DLAB] + [f"{l} (ours)" for l in DLAB],
              ncol=2, fontsize=5.6, handlelength=0.9, columnspacing=0.7,
              handletextpad=0.4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax.grid(axis="y", lw=0.35, color="0.9"); ax.set_axisbelow(True)
    panel_label(ax, "b")

    # ---- c: GC3, real vs de novo ----
    ax = fig.add_subplot(gs[1, 0])
    pos = 0
    xt, xl = [], []
    for k, d in enumerate(DOMS):
        for src in ("real", "generated"):
            v = [100 * x for x in D["gc3"][d][src]]
            bp = ax.boxplot([v], positions=[pos], widths=0.66, showfliers=False,
                            whis=(5, 95), patch_artist=True,
                            medianprops=dict(color="white", lw=0.9),
                            boxprops=dict(linewidth=0.4, edgecolor="0.3"),
                            whiskerprops=dict(linewidth=0.5),
                            capprops=dict(linewidth=0.5))
            bp["boxes"][0].set_facecolor(DOMAIN_C[d])
            bp["boxes"][0].set_alpha(0.9 if src == "real" else 0.38)
            pos += 1
        xt.append(pos - 1.5); xl.append(DLAB[k]); pos += 0.9
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("GC3 content (%)")
    ax.set_ylim(20, 100)
    h = [plt.Rectangle((0, 0), 1, 1, color="0.45", alpha=0.9),
         plt.Rectangle((0, 0), 1, 1, color="0.45", alpha=0.38)]
    ax.legend(h, ["Real (validation)", "De novo generated"], fontsize=6,
              handlelength=0.9, loc="lower center", ncol=2, columnspacing=0.8,
              bbox_to_anchor=(0.5, -0.30))
    ax.grid(axis="y", lw=0.35, color="0.9"); ax.set_axisbelow(True)
    panel_label(ax, "c")

    # ---- d: CDS syntactic validity ----
    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(3); w = 0.34
    tr = [D["cds_validity"][d]["train"] for d in DOMS]
    ge = [D["cds_validity"][d]["generated"] for d in DOMS]
    ax.bar(x - w / 2, tr, width=w * 0.92, color=[DOMAIN_C[d] for d in DOMS],
           alpha=0.85, linewidth=0)
    ax.bar(x + w / 2, ge, width=w * 0.92, color=[DOMAIN_C[d] for d in DOMS],
           alpha=0.35, linewidth=0)
    for xi, (a, b) in enumerate(zip(tr, ge)):
        ax.text(xi - w / 2, a + 1.5, f"{a:.1f}", ha="center", fontsize=5.8, color="0.3")
        ax.text(xi + w / 2, b + 1.5, f"{b:.1f}", ha="center", fontsize=5.8, color="0.3")
    ax.set_xticks(x); ax.set_xticklabels(DLAB)
    ax.set_ylabel("Syntactically valid CDS (%)")
    ax.set_ylim(0, 105)
    h = [plt.Rectangle((0, 0), 1, 1, color="0.45", alpha=0.85),
         plt.Rectangle((0, 0), 1, 1, color="0.45", alpha=0.35)]
    ax.legend(h, ["Training corpus", "De novo generated"], fontsize=6,
              handlelength=0.9, loc="lower left")
    ax.grid(axis="y", lw=0.35, color="0.9"); ax.set_axisbelow(True)
    panel_label(ax, "d")

    save(fig, os.path.join(HERE, "figure2_pretrain"))


if __name__ == "__main__":
    main()
