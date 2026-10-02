"""Length-stratified analysis of the fungal design panel.

The question is whether a generative model trained on natural coding
sequences reproduces the known association between coding-sequence length and
codon optimality, or whether it simply applies the same codon bias at every
length. The four-protein panel in the main text spans 222-676 residues and is
too small to answer that; this panel fixes 40 proteins spanning a much wider
range, 20 synonymous variants each, and reports CAI and the distance to real
high-expression codon usage against protein length.

Output feeds Supplementary Figure S5.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.codon_metrics import cai, load_cai_reference     # noqa: E402
from mrnagpt.vocab import GENETIC_CODE                          # noqa: E402
from sft.evaluate_host_and_realism import (family_profile,      # noqa: E402
                                           score_group)
from sft.compare_pretrained_vs_sft import (group_by_target,     # noqa: E402
                                           read_fasta, to_codons)
from sft.paths import RUNS                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(RUNS / "fungal_sft/generation_length"))
    ap.add_argument("--test-csv", default="sft/data/fungal_expression_test.csv")
    ap.add_argument("--ref-cai", default="sft/lightgbm_expression/cai_reference.json")
    ap.add_argument("--quantile", type=float, default=0.25)
    ap.add_argument("--out", default=str(RUNS / "fungal_sft/"
                                        "generation_length/length_analysis.json"))
    ap.add_argument("--md-out", default="reports/length_panel_analysis.md")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.test_csv)))
    rows.sort(key=lambda r: float(r["Value"]), reverse=True)
    k = max(1, int(len(rows) * args.quantile))
    hi = [to_codons(r["Sequence"]) for r in rows[:k]]
    lo = [to_codons(r["Sequence"]) for r in rows[-k:]]
    prof_hi, prof_lo = family_profile(hi), family_profile(lo)
    gap = score_group(lo, prof_hi, prof_lo)["js_to_high"]
    w = load_cai_reference(args.ref_cai)
    print(f"held-out TEST: {len(rows)} genes -> {k} high / {k} low; "
          f"real high-low gap JS {gap:.6f}", flush=True)

    arms = {"pretrained": "pretrained_eukaryote.fasta",
            "sft_high": "fungal_sft.fasta"}
    out: dict = {"real_high_low_gap_js": gap, "arms": {}}
    for arm, fname in arms.items():
        path = os.path.join(args.panel, fname)
        if not os.path.exists(path):
            print(f"  missing {path}"); continue
        out["arms"][arm] = {}
        for target, seqs in sorted(group_by_target(read_fasta(path)).items()):
            cl = [to_codons(s) for s in seqs.values()]
            r = score_group(cl, prof_hi, prof_lo)
            out["arms"][arm][target] = {
                "n": len(cl),
                "protein_length_aa": len(cl[0]) - 1,     # terminal stop excluded
                "cai_mean": statistics.mean(cai(c, w) for c in cl),
                "cai_sd": statistics.pstdev([cai(c, w) for c in cl]),
                "js_to_high": r["js_to_high"],
                "js_in_gap_units": r["js_to_high"] / gap,
                "llr_high_low": r["llr_high_low"]}
        print(f"  {arm}: {len(out['arms'][arm])} targets", flush=True)

    # real genes are the reference curve: does the model reproduce their trend?
    real = []
    for r in rows[:k]:
        c = to_codons(r["Sequence"])
        real.append({"protein_length_aa": len(c) - 1, "cai": cai(c, w)})
    out["real_high_genes"] = real

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)

    md = ["# Length-stratified design panel", "",
          f"40 target proteins, 20 synonymous variants each, scored against the "
          f"held-out TEST split of the fungal-expression dataset. Real high-low "
          f"gap = {gap:.6f} JS.", "",
          "| length (aa) | arm | CAI | JS to real high (gap units) |",
          "|---:|---|---:|---:|"]
    for arm in out["arms"]:
        for t, v in sorted(out["arms"][arm].items(),
                           key=lambda kv: kv[1]["protein_length_aa"]):
            md.append(f"| {v['protein_length_aa']} | {arm} | {v['cai_mean']:.3f} | "
                      f"{v['js_in_gap_units']:.3f} |")
    open(args.md_out, "w").write("\n".join(md) + "\n")
    print(f"wrote {args.out} and {args.md_out}")


if __name__ == "__main__":
    main()
