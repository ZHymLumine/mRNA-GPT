#!/usr/bin/env python3
"""One overview table over everything pbs/run_stability_eval.sh wrote.

The per-evaluator reports each answer one question well but none of them puts
the three arms (pretrained archaea / high-stability SFT / low-stability control)
next to the two reference rows on one page, which is what actually settles
whether fine-tuning moved codon choice toward measured stability.

The scale matters more than the deltas here and the report says so explicitly:
on this dataset the two REAL profiles are only JS 0.0016 apart (fungal
expression: 0.0398), so a shift has to be read against REAL_low_test rather
than in absolute terms.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sft.paths import RUNS as _RUNS                              # noqa: E402

RUNS = str(_RUNS)
PRE, SFT, LOW = "pretrained_archaea", "stability_sft", "stability_sft_low"


def load(path):
    if not os.path.exists(path):
        print(f"  (missing, skipped) {path}")
        return None
    with open(path) as fh:
        return json.load(fh)


def fmt(v, nd=4):
    return "–" if v is None else f"{v:.{nd}f}"


def realism_block(title, data, low_key, md):
    """One JS/LLR table per metric, three arms plus the two real-gene rows."""
    if not data:
        return
    arms = [(PRE, PRE), (SFT, SFT), ("stability_sft_LOW", LOW)]
    arms = [(k, lab) for k, lab in arms if k in data]
    targets = sorted({t for k, _ in arms for t in data[k]})
    md += [f"## {title}", ""]
    for key, name, nd in (("js_to_high", "JS to the real high-stability profile (lower is better)", 4),
                          ("js_to_low", "JS to the real low-stability profile", 4),
                          ("js_high_minus_low", "JS(high) − JS(low) (negative = more like high stability)", 4),
                          ("llr_high_low", "high/low stability log-likelihood ratio (higher is better)", 4),
                          ("cai_ref", "CAI (high-stability reference table)", 3)):
        md += [f"### {name}", "", "| arm | " + " | ".join(targets) + " |",
               "|---|" + "---:|" * len(targets)]
        for k, lab in arms:
            md.append(f"| {lab} | " + " | ".join(
                fmt(data[k].get(t, {}).get(key), nd) for t in targets) + " |")
        for ref, lab in (("REAL_high_test", "**real high-stability genes (TEST top quartile)**"),
                         ("REAL_low_test", "**real low-stability genes (TEST bottom quartile)**")):
            if ref in data:
                v = data[ref]["_all"].get(key)
                md.append(f"| {lab} | " + " | ".join([f"**{fmt(v, nd)}**"] * len(targets)) + " |")
        md.append("")


def paired_block(title, data, labels, md):
    """CAI/tAI/GC/MFE/LightGBM, paired per target protein."""
    if not data:
        return
    a, b = labels
    if a not in data or b not in data:
        md += [f"## {title}", "", f"(labels {a}/{b} not both present)", ""]
        return
    targets = list(data[a])
    md += [f"## {title}", ""]
    for key, name, nd in (("cai", "CAI", 3), ("tai", "tAI", 3), ("gc", "GC", 4),
                          ("gc3", "GC3", 4), ("mfe_per_nt", "MFE/nt", 4),
                          ("lgbm_predicted_expression", "LightGBM predicted stability", 3)):
        md += [f"### {name}", "", f"| target protein | {a} | {b} | Δ |", "|---|---:|---:|---:|"]
        for t in targets:
            x = data[a][t].get(key, {}).get("mean")
            y = data[b][t].get(key, {}).get("mean")
            d = None if (x is None or y is None) else y - x
            md.append(f"| {t} | {fmt(x, nd)} | {fmt(y, nd)} | "
                      f"{'–' if d is None else f'{d:+.{nd}f}'} |")
        md.append("")
    md += ["### Protein identity and CDS validity", "",
           "| target protein | arm | unique-sequence fraction | pairwise codon identity | nearest-neighbour identity to training set |",
           "|---|---|---:|---:|---:|"]
    for t in targets:
        for lab in (a, b):
            s = data[lab][t].get("synonymous_variants", {})
            ci = s.get("mean_pairwise_codon_identity")
            nn = data[lab][t].get("novelty_best_identity_pct", {}).get("mean")
            md.append(f"| {t} | {lab} | {100*s.get('unique_fraction', 0):.1f}% | "
                      f"{'–' if ci is None else f'{100*ci:.1f}%'} | "
                      f"{'–' if nn is None else f'{nn:.2f}%'} |")
    md.append("")


def uncon_block(title, datas, md):
    """De novo generation: one row per arm, whatever files exist."""
    rows = []
    for data in datas:
        if not data:
            continue
        for lab, d in data.items():
            if lab not in [r[0] for r in rows]:
                rows.append((lab, d))
    if not rows:
        return
    md += [f"## {title}", "",
           "| arm | n | valid CDS rate | CAI | tAI | GC | MFE/nt | LightGBM predicted stability | "
           "nearest-neighbour identity to training set |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for lab, d in rows:
        g = lambda k: d.get(k, {}).get("mean")                       # noqa: E731
        md.append(f"| {lab} | {d.get('n', 0)} | – | {fmt(g('cai'), 3)} | "
                  f"{fmt(g('tai'), 3)} | {fmt(g('gc'), 4)} | {fmt(g('mfe_per_nt'), 4)} | "
                  f"{fmt(g('lgbm_predicted_expression'), 3)} | "
                  f"{fmt(g('novelty_best_identity_pct'), 2)}% |")
    md.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--out", default="reports/stability_sft_summary.md")
    args = ap.parse_args()

    gen = f"{args.runs}/stability_sft/generation"
    panel = f"{args.runs}/stability_sft/generation_panel"

    md = ["# mRNA stability SFT: results overview", "",
          "Three arms: `pretrained_archaea` (not fine-tuned), `stability_sft` (top quartile),",
          "`stability_sft_low` (bottom quartile, negative control). For the upstream "
          "data and splits see",
          "`reports/mrna_stability_leakage.md`, and for the protein-level audit see",
          "`reports/mrna_stability_protein_audit.md`.", "",
          "## ⚠️ Read the scale first", "",
          "Constrained generation can only change codons, so the upper bound on the "
          "movable range is how well codon usage separates high from low stability. "
          "Measured on the held-out TEST split (within synonymous families, "
          "usage-weighted):", "",
          "| dataset | JS(real high-property profile ‖ real low-property profile) | LLR of real low-property genes |",
          "|---|---:|---:|",
          "| fungal expression | 0.03978 | −0.1277 |",
          "| **mRNA stability** | **0.00163** | **−0.0032** |", "",
          "The codon signal in the stability dataset is about 24x weaker than in "
          "expression. In every table below the "
          "`real low-stability genes` row is the **full scale**: an arm's "
          "displacement has to be read against it, not as an absolute value. The "
          "independent predictor likewise only reaches test Pearson 0.376 "
          "(fungal expression: 0.648), so its ranking signal is weak.", ""]

    print("loading artifacts ...")
    realism_block("Target-protein constrained generation: real-gene anchored (no predictor)",
                  load(f"{panel}/stability_realism.json"), LOW, md)
    paired_block("Target-protein constrained generation: paired independent evaluation (fine-tuned vs not)",
                 load(f"{panel}/evaluator_comparison_per_target.json"), (PRE, SFT), md)
    paired_block("Target-protein constrained generation: paired independent evaluation (negative control vs not fine-tuned)",
                 load(f"{panel}/evaluator_comparison_per_target_low.json"), (PRE, LOW), md)
    realism_block("No target protein (de novo): real-gene anchored (no predictor)",
                  load(f"{gen}/stability_realism_unconstrained.json"), LOW, md)
    uncon_block("No target protein (de novo): independent evaluation",
                [load(f"{gen}/evaluator_comparison.json"),
                 load(f"{gen}/evaluator_comparison_low.json")], md)

    md += ["## Protein identity", "",
           "At each step constrained decoding masks the logits to the synonymous "
           "codon set of the current residue, so the translation equals the target "
           "protein **by construction** -- `validate_batch` in "
           "`sft/generate_target_panel.py` re-checks every sequence, the QC output is "
           "in the job logs under "
           "`$MRNA_GPT_RUNS/stability_sft/generation_panel/`, and "
           "`tests/test_generate.py` asserts the same on a randomly initialised model.", ""]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
