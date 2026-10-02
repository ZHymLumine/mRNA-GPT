"""Figure 5. mRNA-GPT compared with established mRNA design methods.

All methods design the same four target proteins under the same
protein-constraint, so amino-acid sequence is fixed and only codon choice
differs. Two independent read-outs are shown because they disagree in an
informative way:

  * how close the codon usage is to real genes at the top of the measured
    property (JS divergence, no trained predictor anywhere);
  * a neural evaluator fine-tuned on held-out measured data.

CAI-maximising methods score well on the neural evaluator only by moving outside
the codon-usage range the evaluator was fit on, which the scatter panels make
explicit.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import C, panel_label, save                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "figure_data.json")))
AM = D["all_methods"]
TARGETS4 = ["Rabies_virus", "Zaire_ebolavirus", "SARS_CoV_2", "Homo_sapiens"]

PRETTY = {
    "pretrained_archaea": "mRNA-GPT (pretrained)", "stability_sft": "mRNA-GPT (fine-tuned)",
    "stability_sft_LOW": "mRNA-GPT (control)",
    "mrna_gpt_pretrained": "mRNA-GPT (pretrained)", "mrna_gpt_sft": "mRNA-GPT (fine-tuned)",
    "mrna_gpt_sft_LOW": "mRNA-GPT (control)",
    "cai_max": "CAI-max", "cai_sample": "CAI-sampling", "gemorna": "GEMORNA",
    "codongpt": "CodonGPT", "icodon": "iCodon", "codonbert_fpp_fix": "CodonBERT",
    "codonbert": "CodonBERT",
    "native_cds": "Native CDS",
}
for lam in ("0", "1", "4", "1p5", "2", "10", "1.5"):
    PRETTY[f"lineardesign_l{lam}"] = f"LinearDesign λ={lam.replace('p', '.')}"
    PRETTY[f"lineardesign_stability_l{lam}"] = f"LinearDesign λ={lam.replace('p', '.')}"
    PRETTY[f"lineardesign_human_l{lam}"] = f"LinearDesign λ={lam.replace('p', '.')} (human)"
    PRETTY[f"ld_fungal_l{lam}"] = f"LinearDesign λ={lam.replace('p', '.')}"
    PRETTY[f"ld_yeast_l{lam}"] = f"LinearDesign λ={lam.replace('p', '.')} (yeast)"


def colour(m):
    if "sft" in m and "LOW" not in m:
        return C["sft"]
    if "LOW" in m:
        return C["sft_low"]
    if "pretrained" in m:
        return C["pretrained"]
    if m == "cai_max":
        return C["cai_max"]
    if m == "cai_sample":
        return C["cai_sample"]
    if m.startswith(("lineardesign", "ld_")):
        return C["ld"]
    return C["other"]


def mean_over_targets(blk, method, key):
    """`key` may be a tuple of aliases: the fungal JSON predates the rename of
    cai_fungal -> cai_ref, and both files are kept as produced."""
    keys = (key,) if isinstance(key, str) else key
    tg = [t for t in blk[method] if t in TARGETS4]
    vals = []
    for t in tg:
        for k in keys:
            if k in blk[method][t]:
                vals.append(blk[method][t][k])
                break
    return float(np.mean(vals)) if vals else np.nan


def per_target(blk, method, key):
    """The value at each of the four targets, in the order of TARGETS4.

    A panel value is the mean of these, so they are what the figure shows as
    the spread around the bar: four targets, not four replicates of one.
    """
    keys = (key,) if isinstance(key, str) else key
    out = []
    for t in TARGETS4:
        if t not in blk.get(method, {}):
            continue
        for k in keys:
            if k in blk[method][t]:
                out.append(blk[method][t][k])
                break
    return out


def spread(values):
    """Mean, sample SD and the values themselves; SD is None below two targets."""
    v = [x for x in values if x is not None and np.isfinite(x)]
    if not v:
        return {"mean": np.nan, "sd": None, "targets": []}
    return {"mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "targets": [float(x) for x in v]}


METHODS = ["mrna_gpt_pretrained", "mrna_gpt_sft", "mrna_gpt_sft_LOW",
           "cai_max",
           "lineardesign_l0", "lineardesign_l1", "lineardesign_l4",
           "codongpt", "gemorna", "codonbert", "native_cds"]
# kept under the old name so nothing that imports it breaks
METHODS13 = METHODS


def realism_table(realism, hi_key="REAL_high_test", lo_key="REAL_low_test"):
    gap = realism[lo_key]["_all"]["js_to_high"]
    # "_selection" records which genes the reference is; it is provenance,
    # not a method
    ms = [m for m in realism
          if not m.startswith("REAL_") and not m.startswith("_")]
    out = {}
    for m in ms:
        js = spread([v / gap for v in per_target(realism, m, "js_to_high")])
        out[m] = {"js": mean_over_targets(realism, m, "js_to_high") / gap,
                  "js_targets": js["targets"], "js_sd": js["sd"],
                  "cai": mean_over_targets(realism, m, ("cai_ref", "cai_fungal"))}
    return out


def oracle_table(oracle, ctx):
    """Per method: the mean over targets the figure plots, and its spread.

    Each target contributes the mean over that target's designs, so the bar is
    a mean of four target means and the error bar is their sample SD.
    """
    out = {}
    for m, per_t in oracle["methods"].items():
        v = [per_t[t][ctx]["mean"] for t in TARGETS4
             if t in per_t and ctx in per_t[t]]
        n = [per_t[t][ctx].get("n") for t in TARGETS4
             if t in per_t and ctx in per_t[t]]
        if v:
            out[m] = spread(v)
            out[m]["n_designs"] = [x for x in n if x]
    return out


def oracle_samples(oracle, ctx):
    """Every scored design of every method, pooled over the four targets.

    The bar-and-whisker form of this column showed a mean over four target
    means; the designs behind it are 200 per target for the seven stochastic
    methods, and the shape of that distribution is what the ridgeline draws.
    A deterministic method has one design per target, so it has four values and
    no distribution -- those are plotted as the four points they are.
    """
    out = {}
    for m, per_t in oracle["methods"].items():
        vals = [v for t in TARGETS4 if t in per_t and ctx in per_t[t]
                for v in per_t[t][ctx]["scores"]]
        if vals:
            out[m] = np.asarray(vals, float)
    return out


def ranked_bar(ax, tab, title, cai_max_train):
    ms = sorted(tab, key=lambda m: tab[m]["js"])
    y = np.arange(len(ms))
    ax.barh(y, [tab[m]["js"] for m in ms], color=[colour(m) for m in ms],
            height=0.68, linewidth=0, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([PRETTY.get(m, m) for m in ms], fontsize=6)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.axvline(1.0, color=C["real_low"], ls="--", lw=0.9, zorder=2)
    ax.axvspan(1e-4, 1.0, color=C["real_high"], alpha=0.10, zorder=0)
    ax.set_xlabel("Jensen–Shannon divergence to\nreal high-property genes")
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(NullLocator())          # decades only
    ax.set_title(title, fontsize=7, pad=8)
    ax.grid(axis="x", lw=0.35, color="0.9"); ax.set_axisbelow(True)
    return ms


def scatter(ax, tab, orc, cai_max_train, title, ylab):
    ms = [m for m in tab if m in orc]
    for m in ms:
        ood = tab[m]["cai"] > cai_max_train
        ax.scatter(tab[m]["js"], orc[m], s=26, color=colour(m),
                   marker="X" if ood else "o",
                   edgecolor="0.25", linewidth=0.4, zorder=3)
    for m in ms:
        if m.startswith(("stability_sft", "mrna_gpt_sft")) and "LOW" not in m \
                or m in ("cai_max", "cai_sample"):
            ax.annotate(PRETTY.get(m, m), (tab[m]["js"], orc[m]), fontsize=5.6,
                        xytext=(-4, 6), textcoords="offset points", color="0.25",
                        ha="right")
    ax.set_xscale("log")
    ax.axvline(1.0, color=C["real_low"], ls="--", lw=0.9, zorder=1)
    ax.axvspan(1e-4, 1.0, color=C["real_high"], alpha=0.07, zorder=0)
    ax.set_xlabel("Jensen–Shannon divergence to\nreal high-property genes")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=7, pad=4)
    ax.grid(lw=0.35, color="0.93"); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], marker="o", ls="", color="0.45", ms=4,
                    label="within evaluator's training range"),
         plt.Line2D([], [], marker="X", ls="", color="0.45", ms=4.5,
                    label="outside it (extrapolation)")]
    ax.legend(handles=h, fontsize=5.6, loc="lower left", handlelength=0.9,
              bbox_to_anchor=(0.0, -0.01))


def main():
    stab = realism_table(AM["stability_realism"], "REAL_high_test", "REAL_low_test")
    expr = realism_table(AM["expression_realism"], "REAL_high_expression_test",
                         "REAL_low_expression_test")
    stab_orc = oracle_table(AM["stability_oracle"], "human_median")
    expr_orc = oracle_table(AM["expression_oracle"], "adh1_yeast")

    fig, axes = plt.subplots(2, 2, figsize=(7.09, 6.4))
    ranked_bar(axes[0, 0], stab, "mRNA stability", 0.908)
    panel_label(axes[0, 0], "a", dx=-0.62)
    ranked_bar(axes[0, 1], expr, "Expression", 0.868)
    panel_label(axes[0, 1], "b", dx=-0.62)

    scatter(axes[1, 0], stab, stab_orc, 0.908, "mRNA stability",
            "Neural evaluator score\n(mRNA-LM, held-out measured data)")
    panel_label(axes[1, 0], "c", dx=-0.28)
    scatter(axes[1, 1], expr, expr_orc, 0.868, "Expression",
            "Neural evaluator score\n(mRNA-LM, held-out measured data)")
    panel_label(axes[1, 1], "d", dx=-0.28)

    fig.tight_layout(w_pad=3.0, h_pad=2.6)
    save(fig, os.path.join(HERE, "figure5_methods"))


if __name__ == "__main__":
    main()
