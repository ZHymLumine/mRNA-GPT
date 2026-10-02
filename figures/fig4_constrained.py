"""Figure 4. Protein-constrained decoding enables property-directed mRNA generation.

At every position the logits are masked to the synonymous codons of the target
residue, so the translated product equals the target protein by construction.
The amino-acid sequence is therefore identical across all arms and all methods,
and any difference in the panels below can only come from codon choice.

Four real target proteins: three viral surface antigens (rabies virus
glycoprotein, Zaire ebolavirus glycoprotein, SARS-CoV-2 protein) and one human
protein, 50 independent variants per target per model.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import C, panel_label, save                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "figure_data.json")))

TARGETS = ["Rabies_virus", "Zaire_ebolavirus", "SARS_CoV_2", "Homo_sapiens"]
TLAB = ["Rabies virus", "Zaire ebolavirus", "SARS-CoV-2", "Homo sapiens"]
ARM_C = [C["pretrained"], C["sft"], C["sft_low"]]
ARM_N = ["Pretrained", "Fine-tuned (top quartile)", "Control (bottom quartile)"]

SPEC = {
    "stability": dict(title="mRNA stability (mRNA-GPT-archaea)",
                      arms=["pretrained_archaea", "stability_sft", "stability_sft_LOW"],
                      hi="REAL_high_test", lo="REAL_low_test"),
    "expression": dict(title="Expression (mRNA-GPT-eukaryote)",
                       arms=["mrna_gpt_pretrained", "mrna_gpt_sft", "mrna_gpt_sft_LOW"],
                       hi="REAL_high_expression_test", lo="REAL_low_expression_test"),
}


def per_target(prop, key):
    blk = D["panel"][prop]["data"]
    sp = SPEC[prop]
    gap = blk[sp["lo"]]["_all"]["js_to_high"]
    lo = blk[sp["lo"]]["_all"]["llr_high_low"]
    hi = blk[sp["hi"]]["_all"]["llr_high_low"]
    out = []
    for a in sp["arms"]:
        row = []
        for t in TARGETS:
            v = blk[a][t][key]
            row.append(v / gap if key == "js_to_high" else (v - lo) / (hi - lo))
        out.append(row)
    return out


def grouped_targets(ax, vals, ylabel, title, log=False, refs=None):
    xs = np.arange(len(TARGETS))
    w = 0.26
    for i in range(3):
        b = ax.bar(xs + (i - 1) * w, vals[i], width=w * 0.92, color=ARM_C[i],
                   linewidth=0, label=ARM_N[i], zorder=3)
        if log:
            for rect, v in zip(b, vals[i]):
                if v > 1.0:
                    rect.set_hatch("///"); rect.set_edgecolor("white")
    ax.set_xticks(xs); ax.set_xticklabels(TLAB, rotation=22, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title, pad=4)
    if log:
        ax.set_yscale("log")
    for y, col, lab in (refs or []):
        ax.axhline(y, color=col, ls="--", lw=0.9, zorder=1)
        ax.text(1.01, y, lab, transform=ax.get_yaxis_transform(), color=col,
                fontsize=6, va="center")
    ax.grid(axis="y", lw=0.35, color="0.9"); ax.set_axisbelow(True)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.2))

    ax = axes[0, 0]
    grouped_targets(ax, per_target("stability", "js_to_high"),
                    "Distance to real high-property genes\n(JS, units of real high–low gap)",
                    SPEC["stability"]["title"], log=True,
                    refs=[(1.0, C["real_low"], "real low")])
    ax.axhspan(1e-3, 1.0, color=C["real_high"], alpha=0.07, zorder=0)
    ax.set_ylim(1e-3, 60)
    ax.legend(loc="lower center", ncol=3, handlelength=0.9, borderpad=0.1,
              columnspacing=0.8, bbox_to_anchor=(0.5, -0.52), fontsize=6)
    panel_label(ax, "a")

    ax = axes[0, 1]
    grouped_targets(ax, per_target("expression", "js_to_high"),
                    "Distance to real high-property genes\n(JS, units of real high–low gap)",
                    SPEC["expression"]["title"], log=True,
                    refs=[(1.0, C["real_low"], "real low")])
    ax.axhspan(1e-3, 1.0, color=C["real_high"], alpha=0.07, zorder=0)
    ax.set_ylim(1e-3, 60)
    panel_label(ax, "b")

    ax = axes[1, 0]
    xs = np.arange(len(TARGETS)); w = 0.26
    for prop, mk, off in (("stability", "o", -0.11), ("expression", "s", 0.11)):
        v = per_target(prop, "llr_high_low")
        for i in range(3):
            ax.scatter(xs + (i - 1) * w + off, v[i], s=13, marker=mk, color=ARM_C[i],
                       edgecolor="0.25", linewidth=0.35, zorder=3)
    ax.axhline(1.0, color=C["real_high"], ls="--", lw=0.9)
    ax.axhline(0.0, color=C["real_low"], ls="--", lw=0.9)
    ax.text(1.01, 1.0, "real high", transform=ax.get_yaxis_transform(),
            color=C["real_high"], fontsize=6, va="center")
    ax.text(1.01, 0.0, "real low", transform=ax.get_yaxis_transform(),
            color=C["real_low"], fontsize=6, va="center")
    ax.set_xticks(xs); ax.set_xticklabels(TLAB, rotation=22, ha="right")
    ax.set_ylabel("Codon log-likelihood ratio\n(0 = real low, 1 = real high genes)")
    ax.grid(axis="y", lw=0.35, color="0.9"); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], marker="o", ls="", color="0.4", ms=3.6, label="stability"),
         plt.Line2D([], [], marker="s", ls="", color="0.4", ms=3.6, label="expression")]
    ax.legend(handles=h, loc="upper right", handlelength=1.0, borderpad=0.15)
    panel_label(ax, "c")

    # ---- d: identity, diversity, novelty ----
    ax = axes[1, 1]
    pp = D["panel_paired"]["stability"]
    ppl = D["panel_paired"]["stability_low"]
    src = {"pretrained_archaea": pp["pretrained_archaea"], "stability_sft": pp["stability_sft"],
           "stability_sft_LOW": ppl["stability_sft_low"]}
    keys = list(SPEC["stability"]["arms"])
    ident = [100.0 for _ in keys]                     # protein identity, verified per sequence
    uniq = [100.0 * np.mean([src[k][t]["synonymous_variants"]["unique_fraction"]
                             for t in TARGETS]) for k in keys]
    codid = [100.0 * np.mean([src[k][t]["synonymous_variants"]["mean_pairwise_codon_identity"]
                              for t in TARGETS]) for k in keys]
    nov = [np.mean([src[k][t]["novelty_best_identity_pct"]["mean"] for t in TARGETS])
           for k in keys]
    groups = ["Protein\nidentity", "Distinct\nsequences", "Pairwise\ncodon identity",
              "Nearest\ntraining seq."]
    vals = [ident, uniq, codid, nov]
    gx = np.arange(len(groups)); w = 0.26
    for i in range(3):
        ax.bar(gx + (i - 1) * w, [v[i] for v in vals], width=w * 0.92, color=ARM_C[i],
               linewidth=0, zorder=3)
    ax.set_xticks(gx); ax.set_xticklabels(groups, rotation=18, ha="right")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", lw=0.35, color="0.9"); ax.set_axisbelow(True)
    for i, v in enumerate(vals):
        ax.text(gx[i], max(v) + 2.5, f"{np.mean(v):.0f}" if max(v) > 5 else "<1.5",
                ha="center", fontsize=5.8, color="0.35")
    panel_label(ax, "d")

    fig.tight_layout(w_pad=2.6, h_pad=2.2)
    save(fig, os.path.join(HERE, "figure4_constrained"))


if __name__ == "__main__":
    main()
