#!/usr/bin/env python3
"""Recompute Figure 5's left column for every method on every task.

One method set, one reference rule, one code path.  The reference genes come
from sft/reference_quartiles.load_split, which both this and
sft/real_gene_reference.py now use, so the divergence denominator and the
descriptor bands of Supplementary Figures S2 and S6 finally describe the same
genes.

The five host-fixed tools (CodonGPT, GEMORNA, iCodon, CodonBERT and the native
coding sequence) carry their codon preference in released weights or in the
source organism, so for a given target protein they emit one design set
whatever the task is.  The same FASTA is therefore scored against all three
references; nothing is regenerated.

Writes figures/figure5_js_all.json.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sft.codon_profile import family_profile, js_divergence  # noqa: E402
from mrnagpt.vocab import GENETIC_CODE                      # noqa: E402
from sft.reference_quartiles import load_split              # noqa: E402

R = os.environ.get("MRNA_GPT_RUNS") or os.path.join(ROOT, "runs")
FB = f"{R}/fungal_sft/generation_panel/baselines"
T4 = ["Rabies_virus", "Zaire_ebolavirus", "SARS_CoV_2", "Homo_sapiens"]

TEST_CSV = {"stability": "sft/data/mrna_stability_test.csv",
            "expression": "sft/data/fungal_expression_test.csv",
            "bacexp": "sft/data/bacteria_expression_test.csv"}

# The five host-fixed tools bake their codon preference into released weights or
# into the source organism, so one design set serves every task; only the
# reference genes change.  native_cds not native_cds_exact (2 targets),
# codonbert_fpp_fix_norm (the length-correct one).
HOST_FIXED = {"codongpt": f"{FB}/codongpt.fasta",
              "gemorna": f"{FB}/gemorna_n200.fasta",
              "codonbert": f"{FB}/codonbert_fpp_fix_norm.fasta",
              "native_cds": f"{FB}/native_cds.fasta"}

PANELS = {
    "stability": dict({
        "mrna_gpt_pretrained": f"{R}/stability_sft/generation_panel_n200/mrna_gpt_pretrained.fasta",
        "mrna_gpt_sft": f"{R}/stability_sft/generation_panel_n200/mrna_gpt_sft.fasta",
        "mrna_gpt_sft_LOW": f"{R}/stability_sft/generation_panel_n200/low/mrna_gpt_sft_LOW.fasta",
        "cai_max": f"{R}/stability_sft/generation_panel/baselines/cai_max.fasta",
        "lineardesign_l0": f"{R}/stability_sft/generation_panel/baselines/lineardesign_stability_l0.fasta",
        "lineardesign_l1": f"{R}/stability_sft/generation_panel/baselines/lineardesign_stability_l1.fasta",
        "lineardesign_l4": f"{R}/stability_sft/generation_panel/baselines/lineardesign_stability_l4.fasta",
    }, **HOST_FIXED),
    "expression": dict({
        "mrna_gpt_pretrained": f"{R}/fungal_sft/generation_panel_n200/pretrained_eukaryote.fasta",
        "mrna_gpt_sft": f"{R}/fungal_sft/generation_panel_n200/fungal_sft.fasta",
        "mrna_gpt_sft_LOW": f"{R}/fungal_sft/generation_panel_low/fungal_sft_low.fasta",
        "cai_max": f"{R}/fungal_sft/generation_panel/cai_max.fasta",
        "lineardesign_l0": f"{FB}/lineardesign_fungal_l0.fasta",
        "lineardesign_l1": f"{FB}/lineardesign_fungal_l1.fasta",
        "lineardesign_l4": f"{FB}/lineardesign_fungal_l4.fasta",
    }, **HOST_FIXED),
    "bacexp": dict({
        # the lr=1e-5 rerun, the one every bacterial result is reported on
        "mrna_gpt_pretrained": f"{R}/bacexp_sweep/generation_panel_n200/mrna_gpt_pretrained.fasta",
        "mrna_gpt_sft": f"{R}/bacexp_sweep/generation_panel_n200/mrna_gpt_sft.fasta",
        "mrna_gpt_sft_LOW": f"{R}/bacexp_sweep/generation_panel_n200/low/mrna_gpt_sft_LOW.fasta",
        "cai_max": f"{R}/bacexp_sft/generation_panel_n200/baselines/cai_max.fasta",
        "lineardesign_l0": f"{R}/bacexp_sft/generation_panel_n200/baselines/lineardesign_bacexp_l0.fasta",
        "lineardesign_l1": f"{R}/bacexp_sft/generation_panel_n200/baselines/lineardesign_bacexp_l1.fasta",
        "lineardesign_l4": f"{R}/bacexp_sft/generation_panel_n200/baselines/lineardesign_bacexp_l4.fasta",
    }, **HOST_FIXED),
}


def read_fasta(path):
    seqs, sid, cur = {}, None, []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if sid is not None:
                seqs[sid] = "".join(cur)
            sid, cur = line[1:], []
        else:
            cur.append(line)
    if sid is not None:
        seqs[sid] = "".join(cur)
    return seqs


def to_codons(s):
    s = s.upper().replace("T", "U")
    return [s[i:i + 3] for i in range(0, len(s) - len(s) % 3, 3)]


def js_to(codon_lists, prof_ref):
    """Usage-weighted divergence of a set of sequences from a reference profile."""
    prof = family_profile(codon_lists)
    usage = Counter()
    for codons in codon_lists:
        for c in codons:
            aa = GENETIC_CODE.get(c)
            if aa and aa != "*":
                usage[aa] += 1
    multi = [aa for aa in prof if len(prof[aa]) > 1]
    tot = sum(usage[aa] for aa in multi) or 1
    return sum(usage[aa] * js_divergence(prof[aa], prof_ref[aa])
               for aa in multi) / tot


def main():
    out = {}
    for task, methods in PANELS.items():
        hi_rows, lo_rows, sel = load_split(TEST_CSV[task])
        prof_high = family_profile([to_codons(r["Sequence"]) for r in hi_rows])
        gap = js_to([to_codons(r["Sequence"]) for r in lo_rows], prof_high)
        out[task] = {"gap": gap, "_selection": sel, "methods": {}}
        print(f"\n=== {task}  gap={gap:.16f}  refs {sel['n_high']}/{sel['n_low']}")
        for m, path in methods.items():
            seqs = read_fasta(path)
            per, n_t = {}, None
            for t in T4:
                sub = [v for k, v in seqs.items() if t in k]
                if sub:
                    per[t] = js_to([to_codons(s) for s in sub], prof_high) / gap
                    n_t = len(sub)
            vals = list(per.values())
            out[task]["methods"][m] = {
                "per_target": per, "mean": st.mean(vals),
                "sd": st.stdev(vals) if len(vals) > 1 else None,
                "n_per_target": n_t, "fasta": path}
            sd = out[task]["methods"][m]["sd"] or 0.0
            print(f"   {m:22s} n={n_t:4d}/tgt  {st.mean(vals):9.3f} +/- {sd:7.3f}")
    with open("figures/figure5_js_all.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote figures/figure5_js_all.json")


if __name__ == "__main__":
    main()
