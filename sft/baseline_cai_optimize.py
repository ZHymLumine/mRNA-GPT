"""Baseline 1: classical CAI optimization (the textbook codon-optimization method).

Reference set = the SFT training subset itself (top-quartile measured expression
of the homology-clean fungal train split), so this baseline sees exactly the same
information the fine-tuned model was given -- the comparison is method vs method,
not data vs data.

Two variants, because they answer different questions:
  - "max"    : per residue, take the highest relative-adaptiveness synonymous
               codon.  Deterministic, one sequence per protein.  This is what
               commercial codon-optimization tools do, and it MAXIMIZES CAI by
               construction -- so CAI is a degenerate metric against it and the
               informative comparisons are MFE / predicted expression / diversity.
  - "sample" : sample codons proportional to their relative adaptiveness, giving
               a batch of variants comparable in size to the model's, and a
               non-degenerate CAI distribution.

Stop codon usage is not part of a CAI table (stops are excluded by definition),
so the terminal stop is drawn from its observed frequency in the same reference
set rather than fixed arbitrarily.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.codon_metrics import build_cai_reference
from mrnagpt.vocab import GENETIC_CODE, STOP_CODONS, codons_for_symbol
from sft.generate_target_panel import load_targets


def to_codons(seq: str) -> list[str]:
    seq = seq.upper().replace("T", "U")
    return [seq[i:i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def load_reference(train_csv: str, quantile: float,
                   threshold: float | None = None) -> tuple[list[list[str]], dict]:
    rows = list(csv.DictReader(open(train_csv)))
    vals = sorted(float(r["Value"]) for r in rows)
    thr = threshold if threshold is not None else vals[int(len(vals) * quantile)]
    kept = [to_codons(r["Sequence"]) for r in rows if float(r["Value"]) >= thr]
    stops = Counter(c[-1] for c in kept if c and c[-1] in STOP_CODONS)
    cut = f"threshold {threshold}" if threshold is not None else f"P{int(quantile*100)}"
    print(f"CAI reference: {len(kept)}/{len(rows)} sequences above {cut} "
          f"({thr:.4f}); stop usage {dict(stops)}")
    return kept, stops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/protein_sequences.csv")
    ap.add_argument("--train-csv", default="sft/data/fungal_expression_train.csv")
    ap.add_argument("--quantile", type=float, default=0.75,
                    help="same top-quartile cut used to build the SFT training set")
    ap.add_argument("--threshold", type=float, default=None,
                    help="absolute cut on Value instead of a quantile, to mirror "
                         "--threshold in sft/prepare_sft_data.py (e.g. 2 for TE). The "
                         "baseline must see exactly the reference set the model was "
                         "fine-tuned on, or the comparison is data vs data.")
    ap.add_argument("--n", type=int, default=50, help="variants per target for --mode sample")
    ap.add_argument("--mode", choices=["max", "sample"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ref_codon_lists, stops = load_reference(args.train_csv, args.quantile,
                                            args.threshold)
    w = build_cai_reference(ref_codon_lists)
    stop_codon = max(stops, key=stops.get) if stops else "UAA"

    # per amino acid: synonymous codons ordered by relative adaptiveness
    by_aa: dict[str, list[str]] = defaultdict(list)
    for c, aa in GENETIC_CODE.items():
        if aa != "*" and c in w:
            by_aa[aa].append(c)
    for aa in by_aa:
        by_aa[aa].sort(key=lambda c: -w[c])

    rng = random.Random(args.seed)
    label = f"cai_{args.mode}"
    n = 1 if args.mode == "max" else args.n
    written = 0
    with open(args.out, "w") as fh:
        for target, prot in load_targets(args.csv):
            for i in range(n):
                codons = []
                for aa in prot:
                    cands = by_aa.get(aa) or list(codons_for_symbol(aa))
                    if args.mode == "max":
                        codons.append(cands[0])
                    else:
                        weights = [w.get(c, 0.0) for c in cands]
                        if sum(weights) <= 0:
                            weights = [1.0] * len(cands)
                        codons.append(rng.choices(cands, weights=weights, k=1)[0])
                codons[0] = "AUG" if prot[0] == "M" else codons[0]
                codons.append(stop_codon)
                fh.write(f">{label}|{target}|{i} n_codon={len(codons)}\n{''.join(codons)}\n")
                written += 1
    meta = {"mode": args.mode, "stop_codon": stop_codon, "quantile": args.quantile,
            "n_reference_seqs": len(ref_codon_lists), "n_written": written}
    with open(args.out + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {written} sequences -> {args.out}")


if __name__ == "__main__":
    main()
