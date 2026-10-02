"""Protein-constrained generation over a panel of real target proteins.

Extends the earlier single-target comparison (one held-out fungal test protein)
to the panel in data/protein_sequences.csv -- four real targets of independent
interest (three viral surface antigens plus a human protein).  Each target gets
N independent stochastic completions from BOTH checkpoints, so every metric
downstream is a *paired* comparison at fixed amino-acid sequence: the two models
differ only in codon choice, never in which protein was generated.

FASTA ids are "{label}|{target}|{i}", which is what
sft/compare_pretrained_vs_sft.py --per-target groups on.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrnagpt.generate import constrained_sample, load_model, validate_batch  # noqa: E402
from sft.paths import RUNS                                                   # noqa: E402

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def load_targets(csv_path: str) -> list[tuple[str, str]]:
    out = []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            seq = row["sequence"].strip().upper()
            bad = sorted(set(seq) - STD_AA)
            if bad:
                raise ValueError(f"{row['organism']}: non-standard residues {bad} "
                                 "(the synonymous-codon mask has no entry for these)")
            out.append((slug(row["organism"]), seq))
    if len(out) != len({s for s, _ in out}):
        raise ValueError("duplicate organism names -> ambiguous FASTA group ids")
    return out


def run_one(label: str, ckpt: str, targets, n: int, args) -> dict:
    model = load_model(ckpt, args.device)
    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed)

    fasta_path = os.path.join(args.out_dir, f"{label}.fasta")
    per_target = {}
    with open(fasta_path, "w") as fh:
        for target, prot in targets:
            outs = constrained_sample(model, [prot] * n, temperature=args.temperature,
                                      top_k=args.top_k, top_p=args.top_p,
                                      device=args.device, generator=gen,
                                      batch_size=args.batch_size)
            rows, md = validate_batch([(o, prot) for o in outs], f"{label} / {target}")
            print(md, flush=True)
            per_target[target] = {
                "n": len(rows),
                "protein_len": len(prot),
                "valid_cds_pct": 100.0 * sum(r["valid_cds"] for r in rows) / len(rows),
                "protein_match_pct": 100.0 * sum(bool(r["protein_match"]) for r in rows) / len(rows),
                "mean_codon_len": sum(r["n_codon"] for r in rows) / len(rows),
                "mean_gc": sum(r["gc_content"] for r in rows) / len(rows),
                "mean_gc3": sum(r["gc3"] for r in rows) / len(rows),
            }
            for i, codons in enumerate(outs):
                fh.write(f">{label}|{target}|{i} n_codon={len(codons)}\n{''.join(codons)}\n")
    del model
    torch.cuda.empty_cache()
    print(f"wrote {n * len(targets)} sequences -> {fasta_path}", flush=True)
    return {"label": label, "ckpt": ckpt, "fasta": fasta_path, "per_target": per_target}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/protein_sequences.csv")
    ap.add_argument("--pretrained-ckpt", default=str(RUNS / "eukaryote/model_best.pt"))
    ap.add_argument("--sft-ckpt", default=str(RUNS / "fungal_sft/model_best.pt"))
    ap.add_argument("--out-dir", default=str(RUNS / "fungal_sft/generation_panel"))
    ap.add_argument("--n", type=int, default=50, help="samples per target per model")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=16, help="KV cache scales with this x sequence length")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sft-label", default="fungal_sft",
                    help="FASTA label for the fine-tuned run (e.g. fungal_sft_low for the "
                         "negative control, so its sequences group separately downstream)")
    ap.add_argument("--pretrained-label", default="pretrained_eukaryote",
                    help="FASTA label for the pretrained checkpoint (e.g. pretrained_archaea)")
    ap.add_argument("--only", choices=["both", "sft", "pretrained"], default="both",
                    help="skip re-generating a checkpoint that has already been sampled")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    targets = load_targets(args.csv)
    print(f"panel: {[(t, len(p)) for t, p in targets]}", flush=True)

    manifest = {"panel_csv": args.csv, "n_per_target": args.n, "seed": args.seed,
                "temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
                "targets": {t: {"protein_len": len(p), "protein": p} for t, p in targets},
                "runs": {}}
    runs = [(args.pretrained_label, args.pretrained_ckpt), (args.sft_label, args.sft_ckpt)]
    if args.only == "sft":
        runs = runs[1:]
    elif args.only == "pretrained":
        runs = runs[:1]
    for label, ckpt in runs:
        manifest["runs"][label] = run_one(label, ckpt, targets, args.n, args)

    path = os.path.join(args.out_dir, "generation_manifest.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
