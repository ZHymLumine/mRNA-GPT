"""Per-target evaluation across an arbitrary set of methods (models and baselines).

sft/compare_pretrained_vs_sft.py compares exactly two checkpoints; this takes
N labelled FASTAs -- mRNA-GPT before/after SFT, classical CAI optimization,
LinearDesign at several lambdas, GEMORNA -- and reports every evaluator per
target protein, methods as columns.

Adds one column the two-checkpoint script did not need: protein identity.
mRNA-GPT's synonymous-codon masking makes it exact by construction and classical
CAI optimization is exact by definition, but a seq2seq generator has no such
guarantee, so identity has to be measured rather than assumed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.codon_metrics import load_cai_reference, load_tai_weights
from evaluate.lightgbm_expression import load_predictor
from mrnagpt.generate import validate_cds
from sft.compare_pretrained_vs_sft import (evaluate_seqs, group_by_target,
                                           load_fungal_train_reference, read_fasta,
                                           to_codons)
from sft.generate_target_panel import load_targets

def metrics_for(property_name: str):
    return [("protein_match_pct", "protein identity, per residue %", 1),
            ("valid_cds_pct", "valid CDS %", 1),
            ("cai", "CAI", 3), ("tai", "tAI", 3), ("gc3", "GC3", 3),
            ("mfe_per_nt", "MFE/nt", 4),
            ("lgbm_predicted_expression", f"LightGBM predicted {property_name}", 2),
            ("novelty_best_identity_pct", "nearest-neighbour identity to training set %", 2),
            ("mean_pairwise_codon_identity", "pairwise codon identity", 3),
            ("n", "n", 0)]


def cds_checks(seqs: dict[str, str], protein: str) -> dict:
    rows = [validate_cds(to_codons(s), protein) for s in seqs.values()]
    n = max(len(rows), 1)
    return {"valid_cds_pct": 100.0 * sum(r["valid_cds"] for r in rows) / n,
            "protein_match_pct": 100.0 * sum(bool(r["protein_match"]) for r in rows) / n}


def get_metric(res: dict, key: str):
    if key == "n":
        return res["n"]
    if key in ("protein_match_pct", "valid_cds_pct"):
        return res[key]
    if key == "mean_pairwise_codon_identity":
        return res["synonymous_variants"].get("mean_pairwise_codon_identity")
    return res[key].get("mean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", action="append", required=True, metavar="LABEL=PATH",
                    help="repeatable; label is the column name in the report")
    ap.add_argument("--panel-csv", default="data/protein_sequences.csv")
    ap.add_argument("--lgbm-dir", default="sft/lightgbm_expression")
    ap.add_argument("--train-csv", default="sft/data/fungal_expression_train.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md-out", required=True)
    ap.add_argument("--property-name", default="expression",
                    help="what --lgbm-dir predicts, for the table heading")
    args = ap.parse_args()

    methods = []
    for spec in args.fasta:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--fasta needs LABEL=PATH, got {spec!r}")
        if not os.path.exists(path):
            raise SystemExit(f"{label}: no such FASTA {path}")
        methods.append((label, path))

    proteins = dict(load_targets(args.panel_csv))
    cai_ref = load_cai_reference(f"{args.lgbm_dir}/cai_reference.json")
    tai_w = load_tai_weights()
    lgbm_model, lgbm_cai_ref = load_predictor(args.lgbm_dir)
    reference_seqs = load_fungal_train_reference(args.train_csv)
    ev_args = (cai_ref, tai_w, lgbm_model, lgbm_cai_ref, reference_seqs)

    results: dict = {}
    for label, path in methods:
        results[label] = {}
        for target, sub in group_by_target(read_fasta(path)).items():
            if target not in proteins:
                raise SystemExit(f"{label}: target {target!r} is not in {args.panel_csv}")
            print(f"evaluating {label} / {target} (n={len(sub)}) ...", flush=True)
            res = evaluate_seqs(f"{label}|{target}", sub, *ev_args)
            res.update(cds_checks(sub, proteins[target]))
            results[label][target] = res

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {args.out}")

    targets = list(proteins)
    labels = [lab for lab, _ in methods]
    md = ["# Target-protein panel: mRNA-GPT against the baselines, protein by protein", "",
          f"All methods are compared on the same set of target proteins under the same "
          f"independent evaluators. The CAI reference set and the LightGBM "
          f"{args.property_name} predictor are both built from this dataset's train "
          f"split only (`--lgbm-dir {args.lgbm_dir}`, `--train-csv {args.train_csv}`).", ""]
    for key, name, nd in metrics_for(args.property_name):
        md += [f"## {name}", "", "| target protein | " + " | ".join(labels) + " |",
               "|---|" + "---:|" * len(labels)]
        for t in targets:
            cells = []
            for lab in labels:
                v = get_metric(results[lab][t], key) if t in results[lab] else None
                cells.append("–" if v is None else f"{v:.{nd}f}")
            md.append(f"| {t} | " + " | ".join(cells) + " |")
        md.append("")
    with open(args.md_out, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()
