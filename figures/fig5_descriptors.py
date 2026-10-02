"""Sequence descriptors (CAI, tAI, GC3, MFE) for every method, against what real
genes actually do -- panels e-h of Figure 5.

The point of these panels is NOT to rank methods. None of these four quantities
is a quality metric with a universal direction:

  * real top-quartile genes sit at a FINITE CAI (0.68 / 0.77 / 0.80 across the
    three tasks), not at 1.0, so "higher CAI" stops meaning "better" well before
    the maximum. CAI-max reaches 0.97-1.00, up to 43% above any real
    high-property gene.
  * GC3 has no direction at all; it is a host property (our own pretraining
    corpora: archaea 73%, bacteria 60%, eukaryote 51% median GC3).
  * MFE's desired direction depends on the objective -- stronger structure
    resists hydrolysis but 5'-proximal structure blocks initiation.
  * and in the bacterial-expression task the sign is simply inverted: real
    high-expression genes have LOWER CAI (0.801 vs 0.804) and LOWER GC3
    (0.542 vs 0.626) than real low-expression ones.

So each panel is drawn as a distance-to-real-genes plot: the shaded band is the
interval spanned by real top- and bottom-quartile genes, and what matters is
whether a method lands inside it.
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
T4 = ["Rabies_virus", "Zaire_ebolavirus", "SARS_CoV_2", "Homo_sapiens"]
REAL = json.load(open(os.path.join(HERE, "real_gene_metrics.json")))

TASKS = [("expression", "Fungal transcript expression",
          f"{R}/fungal_sft/generation_panel_n200/all_methods_property_eval_aligned.json"),
         ("stability", "mRNA stability",
          f"{R}/stability_sft/generation_panel_n200/all_methods_property_eval_full.json"),
         # the bacterial panel is the swept re-run; runs/bacexp_sft is superseded
         ("bacexp", "Bacteria protein expression",
          f"{R}/bacexp_sweep/generation_panel_n200/all_methods_property_eval.json")]

METRICS = [("cai", "CAI", 3), ("tai", "tAI", 3),
           ("gc3", "GC3", 3), ("mfe_per_nt", "MFE per nt", 3)]

# One canonical row order shared by every panel, so a given row means the same
# method in all of them. Sorting each column independently (by CAI, say) would
# be more legible per panel but silently mislabels every column that does not
# carry the tick labels.
ORDER = [("mrna_gpt_pretrained", "mRNA-GPT (pretrained)"),
         ("mrna_gpt_sft", "mRNA-GPT (fine-tuned)"),
         ("mrna_gpt_sft_LOW", "mRNA-GPT (low control)"),
         ("native_cds", "Native CDS"),
         ("cai_max", "CAI-max"),
         ("cai_sample", "CAI-sampling"),
         # LinearDesign as a user actually gets it: the codon-usage table shipped
         # with the tool, host-matched per task (yeast for fungal expression,
         # human for stability). The variant re-fitted on each task's own
         # high-property genes was dropped -- it is not a method anyone can run
         # off the shelf. No bacterial table ships, so the bacterial panel has
         # no LinearDesign row.
         ("lineardesign_yeast_l0", "LinearDesign \u03bb=0"),
         ("lineardesign_human_l0", "LinearDesign \u03bb=0"),
         ("lineardesign_yeast_l1", "LinearDesign \u03bb=1"),
         ("lineardesign_human_l1", "LinearDesign \u03bb=1"),
         ("lineardesign_yeast_l4", "LinearDesign \u03bb=4"),
         ("lineardesign_human_l4", "LinearDesign \u03bb=4"),
         ("codongpt", "CodonGPT"), ("gemorna", "GEMORNA"),
         ("icodon", "iCodon"), ("codonbert", "CodonBERT")]
# the yeast/human pairs occupy one row each: no task has both
ROWS, seen = [], set()
for k, lab in ORDER:
    if lab in seen:
        ROWS[[r[1] for r in ROWS].index(lab)][0].append(k)
    else:
        ROWS.append(([k], lab))
        seen.add(lab)


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


def pick(tab, keys):
    for k in keys:
        if k in tab:
            return k
    return None


def cell(d, m, t, k):
    v = d[m][t]
    return v[k]["mean"] if isinstance(v.get(k), dict) else v.get(k)


def table(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    out = {}
    for m in d:
        ts = [t for t in T4 if t in d[m] and (cell(d, m, t, "protein_match_pct") or 100) >= 100]
        if not ts:
            continue
        out[m] = {k: float(np.mean([cell(d, m, t, k) for t in ts])) for k, _, _ in METRICS}
        out[m]["n"] = d[m][ts[0]]["n"]
    return out


def main():
    tabs = [(key, lab, table(p)) for key, lab, p in TASKS]
    have = [(k, l, t) for k, l, t in tabs if t]
    missing = [l for k, l, t in tabs if not t]

    fig, axes = plt.subplots(len(METRICS), len(have), figsize=(2.9 * len(have) + 1.0, 9.2),
                             squeeze=False)
    for col, (key, label, tab) in enumerate(have):
        hi, lo = REAL[key]["real_high"], REAL[key]["real_low"]
        keys = [pick(tab, ks) for ks, _ in ROWS]
        # Does the real-gene band transfer to this panel? The reference genes are
        # real fungal / archaeal / E. coli CDSs; the panel designs viral and human
        # proteins. CAI and tAI are weighted by amino-acid composition, so a
        # different protein shifts the achievable range even at identical codon
        # preferences. CAI-sampling is the composition-matched control -- it
        # applies the real high-property codon usage to the panel's OWN proteins --
        # so if it lands further from the real high mean than the whole real
        # high-low gap, the band does not transfer for that metric and is drawn
        # hatched rather than solid.
        cs = tab.get(pick(tab, ["cai_sample"]))
        transfers = {m: (cs is None or abs(cs[m] - hi[m]) <= abs(hi[m] - lo[m]))
                     for m, _, _ in METRICS}
        for row, (mkey, mlabel, nd) in enumerate(METRICS):
            ax = axes[row][col]
            y = np.arange(len(ROWS))
            band = sorted((hi[mkey], lo[mkey]))
            ok = transfers[mkey]
            ax.axvspan(band[0], band[1], color=C["real_high"],
                       alpha=0.18 if ok else 0.07, zorder=0)
            ax.axvline(hi[mkey], color=C["real_high"], ls="--",
                       lw=0.9 if ok else 0.6, alpha=1.0 if ok else 0.4, zorder=2)
            ax.axvline(lo[mkey], color=C["real_low"], ls=":",
                       lw=0.9 if ok else 0.6, alpha=1.0 if ok else 0.4, zorder=2)
            if not ok:
                # these bands are only a few thousandths wide, so a hatch does
                # not read at print size; label the panel instead
                ax.text(0.5, 1.005, "reference not comparable",
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=5.4, color="0.45", style="italic")
            for i, k in enumerate(keys):
                if k is None:
                    # No bar for this method in this task. An empty row reads as
                    # "zero" or "we forgot", so say which it is: LinearDesign
                    # ships no bacterial codon table, so it cannot appear on the
                    # bacterial panel at all.
                    # x in axes fraction (the limits are not settled yet),
                    # y in data coordinates
                    ax.text(0.02, i, "not available",
                            transform=ax.get_yaxis_transform(),
                            ha="left", va="center", fontsize=4.8,
                            color="0.6", style="italic", zorder=3)
                    continue
                ax.barh(i, tab[k][mkey], color=colour(k), height=0.68,
                        linewidth=0, zorder=3)
            ax.set_yticks(y)
            ax.set_yticklabels([lab for _, lab in ROWS] if col == 0 else [], fontsize=5.2)
            for i, k in enumerate(keys):            # grey out absent methods
                if k is None and col == 0:
                    ax.get_yticklabels()[i].set_color("0.75")
            ax.invert_yaxis()
            ax.tick_params(axis="x", labelsize=5.6)
            ax.set_ylim(len(ROWS) - 0.4, -0.6)
            if row == 0:
                ax.set_title(label, fontsize=7.5, pad=12 if not ok else 3)
            if col == 0:
                ax.set_ylabel(mlabel, fontsize=7.5)
            ax.grid(axis="x", lw=0.3, color="0.92")
            ax.set_axisbelow(True)

    h = [plt.Line2D([], [], color=C["real_high"], ls="--", lw=1, label="real top-quartile genes"),
         plt.Line2D([], [], color=C["real_low"], ls=":", lw=1, label="real bottom-quartile genes"),
         plt.Rectangle((0, 0), 1, 1, color=C["real_high"], alpha=0.18,
                       label="range spanned by real genes"),
         plt.Rectangle((0, 0), 1, 1, color=C["real_high"], alpha=0.07,
                       label="reference not comparable")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, -0.012))
    for i, lab in enumerate("abcd"):
        panel_label(axes[i][0], lab, dx=-0.92, dy=1.06)
    fig.tight_layout(rect=(0, 0.022, 1, 1), h_pad=1.4, w_pad=1.0)
    save(fig, os.path.join(HERE, "figure5_descriptors"))

    tsv = os.path.join(HERE, "figure5_descriptors.tsv")
    with open(tsv, "w") as fh:
        fh.write("task\tmethod\tn\t" + "\t".join(k for k, _, _ in METRICS) +
                 "\tin_real_range_of_transferable\n")
        for key, label, tab in have:
            hi, lo = REAL[key]["real_high"], REAL[key]["real_low"]
            for ks, mlab in ROWS:
                k = pick(tab, ks)
                if k is None:
                    continue
                cs = tab.get(pick(tab, ["cai_sample"]))
                ok = {mk: (cs is None or
                           abs(cs[mk] - hi[mk]) <= abs(hi[mk] - lo[mk]))
                      for mk, _, _ in METRICS}
                inside = sum(ok[mk] and
                             min(hi[mk], lo[mk]) <= tab[k][mk] <= max(hi[mk], lo[mk])
                             for mk, _, _ in METRICS)
                fh.write(f"{key}\t{mlab}\t{tab[k]['n']}\t" +
                         "\t".join(f"{tab[k][mk]:.4f}" for mk, _, _ in METRICS) +
                         f"\t{inside}/{sum(ok.values())}\n")
            for ref in ("real_high", "real_low"):
                fh.write(f"{key}\t[{ref} genes]\t200\t" +
                         "\t".join(f"{REAL[key][ref][mk]:.4f}" for mk, _, _ in METRICS) +
                         "\t-\n")
    print("wrote " + tsv)
    if missing:
        print("pending (job still queued): " + ", ".join(missing))


if __name__ == "__main__":
    main()
