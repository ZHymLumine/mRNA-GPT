"""Host-symmetric and ground-truth-anchored evaluation of every design method.

Two problems with the first cross-method table this fixes:

1. HOST ASYMMETRY. CAI was only ever computed against our fungal reference, so
   every method that was given fungal information scored well and GEMORNA --
   which cannot be re-hosted, its codon preference is baked into released
   weights -- scored badly for a reason that is not about design quality. Here
   CAI is reported under BOTH a fungal and a human reference, and each method is
   labelled with the host information it was actually given.

2. EVALUATOR CIRCULARITY. The LightGBM predictor takes CAI as an input feature
   (19.6% of total gain) and correlates with CAI at r=0.88 over generated
   sequences, so ranking CAI-maximizing methods by it is close to a tautology.
   The metrics added here use no trained predictor at all: they compare a
   method's synonymous codon choices against the EMPIRICAL choices of real
   high-property genes, measured on the held-out TEST split -- unseen by SFT
   training, by the CAI reference (train top-10%) and by LightGBM (train).

Property-agnostic via --test-csv / --property-label: the same code scores the
fungal-expression dataset and the mRNA-stability dataset. --alt-cai-table can be
set to '' when every method being compared was given the same host information,
in which case only the reference host's CAI is reported.

Codon usage is compared WITHIN each synonymous family (p(codon | amino acid)),
so amino-acid composition -- which differs between the target proteins and the
dataset's genes, and which no design method controls -- cannot influence the
score.

  js_to_high : Jensen-Shannon divergence to the real high-property profile,
               usage-weighted over families. Lower = more like a real gene at
               the top of the measured property. This is the metric that
               penalizes OVERSHOOT: picking one codon per family drives every
               family's entropy to zero, which no natural gene does.
  llr_high_low : mean per-codon log(p_high / p_low). Higher = each codon choice
               is more characteristic of high- than low-property genes. This
               one does reward extremes, so the two are reported together.

Real test-split genes are scored the same way and reported as reference rows:
they are what "a real gene at the top of this property" scores.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sft.paths import external                                   # noqa: E402

from evaluate.codon_metrics import cai, load_cai_reference
from sft.reference_quartiles import load_split
from mrnagpt.vocab import GENETIC_CODE
from sft.codon_profile import (PSEUDO, family_profile,  # noqa: F401
                               js_divergence, score_group)
from sft.compare_pretrained_vs_sft import group_by_target, read_fasta, to_codons

def load_human_cai_reference(path: str) -> dict:
    """LinearDesign's human codon-usage CSV -> relative adaptiveness w = f / max(f)."""
    by_aa: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != 3 or parts[0].startswith("#"):
                continue
            codon, aa, freq = parts[0].strip(), parts[1].strip(), float(parts[2])
            if aa != "*":
                by_aa[aa][codon] = freq
    w = {}
    for aa, d in by_aa.items():
        m = max(d.values())
        for c, f in d.items():
            w[c] = f / m if m > 0 else 0.0
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--test-csv", default="sft/data/fungal_expression_test.csv")
    ap.add_argument("--quantile", type=float, default=0.25,
                    help="top/bottom fraction of the TEST split defining high/low property")
    ap.add_argument("--ref-cai", "--fungal-cai", dest="ref_cai",
                    default="sft/lightgbm_expression/cai_reference.json",
                    help="CAI reference built from the TRAIN top-10% of this dataset")
    ap.add_argument("--ref-cai-label", default="fungal")
    ap.add_argument("--alt-cai-table", "--human-table", dest="alt_cai_table",
                    default=str(external("LinearDesign", "LINEARDESIGN_DIR") /
                                "codon_usage_freq_table_human.csv"),
                    help="a SECOND host's codon-usage table (LinearDesign CSV format), so "
                         "methods that cannot be re-hosted are not penalised by a "
                         "single-host CAI. Pass '' to report only the reference host.")
    ap.add_argument("--alt-cai-label", default="human")
    ap.add_argument("--property-label", default="expression",
                    help="what the TEST-split Value measures, for the table headings "
                         "(e.g. stability)")
    ap.add_argument("--ungrouped", action="store_true",
                    help="treat each FASTA as ONE group (for de novo / unconstrained "
                         "generation, whose ids carry no target field). Within-family "
                         "codon usage is still the comparison, so the metric means the "
                         "same thing: given whatever amino-acid composition the model "
                         "chose, are its synonymous choices those of real high-property "
                         "genes.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md-out", required=True)
    args = ap.parse_args()
    prop = args.property_label

    hi_rows, lo_rows, selection = load_split(args.test_csv, args.quantile)
    high = [to_codons(r["Sequence"]) for r in hi_rows]
    low = [to_codons(r["Sequence"]) for r in lo_rows]
    print(f"held-out TEST split: {selection['n_total']} genes -> {len(high)} high "
          f"/ {len(low)} low, value >= {selection['high_threshold']:.6g} and "
          f"<= {selection['low_threshold']:.6g}", flush=True)
    prof_high, prof_low = family_profile(high), family_profile(low)

    cai_tables = {"cai_ref": load_cai_reference(args.ref_cai)}
    cai_labels = {"cai_ref": f"CAI ({args.ref_cai_label} table)"}
    host_labels = {"cai_ref": args.ref_cai_label}
    if args.alt_cai_table:
        cai_tables["cai_alt"] = load_human_cai_reference(args.alt_cai_table)
        cai_labels["cai_alt"] = f"CAI ({args.alt_cai_label} table)"
        host_labels["cai_alt"] = args.alt_cai_label

    def add_cai(r: dict, codon_lists) -> dict:
        for key, w in cai_tables.items():
            r[key] = statistics.mean(cai(c, w) for c in codon_lists)
        return r

    def cai_str(r: dict) -> str:
        return " ".join(f"{host_labels[k]} {r[k]:.3f}" for k in cai_tables)

    results: dict = {}
    for spec in args.fasta:
        label, _, path = spec.partition("=")
        if not os.path.exists(path):
            raise SystemExit(f"{label}: no such FASTA {path}")
        results[label] = {}
        seqs_all = read_fasta(path)
        groups = {"all": seqs_all} if args.ungrouped else group_by_target(seqs_all)
        for target, sub in groups.items():
            cl = [to_codons(s) for s in sub.values()]
            r = add_cai(score_group(cl, prof_high, prof_low), cl)
            results[label][target] = r
            print(f"{label}/{target}: JS {r['js_to_high']:.4f} LLR {r['llr_high_low']:+.4f} "
                  f"{cai_str(r)}", flush=True)

    # reference rows: what real genes score under the same measurement
    ref_rows = (("REAL_high_test", high), ("REAL_low_test", low))
    for name, group in ref_rows:
        r = add_cai(score_group(group, prof_high, prof_low), group)
        results[name] = {"_all": r}
        print(f"{name}: JS {r['js_to_high']:.4f} LLR {r['llr_high_low']:+.4f} "
              f"{cai_str(r)}", flush=True)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)

    labels = [s.partition("=")[0] for s in args.fasta]
    targets = sorted({t for lab in labels for t in results[lab]})
    hosts = " and ".join(host_labels[k] for k in cai_tables)
    md = ["# Host-symmetric, real-gene-anchored comparison", "",
          f"CAI is reported against the {hosts} reference table(s); JS/LLR are built "
          f"from the real measured {prop} values of the held-out TEST split, with no "
          f"trained predictor in the loop. Lower JS = more like a real high-{prop} "
          f"gene; higher LLR = codon choice leans more toward high-{prop} than "
          f"low-{prop} genes.", "",
          f"The reference rows `REAL_high_test` / `REAL_low_test` are what the real "
          f"genes in the top / bottom {int(100*args.quantile)}% of {prop} in the "
          f"held-out TEST split score under the same measurement.", ""]
    rows = [("js_to_high", f"JS to the real high-{prop} profile (lower is better)", 4),
            ("js_to_low", f"JS to the real low-{prop} profile", 4),
            ("js_high_minus_low", "JS(high) − JS(low) (negative = more like high "
             + prop + ")", 4),
            ("llr_high_low", f"high/low {prop} log-likelihood ratio (higher is better)", 4)]
    rows += [(k, cai_labels[k], 3) for k in cai_tables]
    for key, name, nd in rows:
        md += [f"## {name}", "", "| method | " + " | ".join(targets) + " |",
               "|---|" + "---:|" * len(targets)]
        for lab in labels:
            cells = [f"{results[lab][t][key]:.{nd}f}" if t in results[lab] else "–"
                     for t in targets]
            md.append(f"| {lab} | " + " | ".join(cells) + " |")
        for ref, _ in ref_rows:
            v = results[ref]["_all"][key]
            md.append(f"| **{ref}** | " + " | ".join([f"**{v:.{nd}f}**"] * len(targets)) + " |")
        md.append("")
    with open(args.md_out, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"wrote {args.out} and {args.md_out}")


if __name__ == "__main__":
    main()
