"""Normalize an external tool's FASTA into the panel convention used downstream.

Tools differ in what they emit: CodonBERT (FPPGroup) writes one deterministic
sequence per protein with the FASTA header set to the protein name and no
terminal stop codon; LinearDesign also omits the stop. Downstream evaluation
groups sequences by the target encoded in the header ("{label}|{target}|{i}")
and counts a CDS without a stop codon as invalid, which would penalize a method
for a convention rather than for its design quality.

This rewrites headers and, when --add-stop is given, appends the same stop codon
used for every other method (the most frequent stop in the top-quartile fungal
reference set), recording that it was added.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrnagpt.vocab import GENETIC_CODE, STOP_CODONS


def read_fasta(path: str) -> list[tuple[str, str]]:
    out, name, cur = [], None, []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(cur)))
            name, cur = line[1:].split()[0], []
        else:
            cur.append(line)
    if name is not None:
        out.append((name, "".join(cur)))
    return out


def reference_stop(train_csv: str, quantile: float) -> str:
    rows = list(csv.DictReader(open(train_csv)))
    vals = sorted(float(r["Value"]) for r in rows)
    thr = vals[int(len(vals) * quantile)]
    counts: Counter = Counter()
    for r in rows:
        if float(r["Value"]) >= thr:
            s = r["Sequence"].upper().replace("T", "U")
            cod = [s[i:i + 3] for i in range(0, len(s) - len(s) % 3, 3)]
            if cod and cod[-1] in STOP_CODONS:
                counts[cod[-1]] += 1
    return max(counts, key=counts.get) if counts else "UAA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-fasta", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--panel-csv", default="data/protein_sequences.csv")
    ap.add_argument("--train-csv", default="sft/data/fungal_expression_train.csv")
    ap.add_argument("--quantile", type=float, default=0.75)
    ap.add_argument("--add-stop", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    known = {re.sub(r"[^A-Za-z0-9]+", "_", r["organism"]).strip("_"): r["sequence"].strip().upper()
             for r in csv.DictReader(open(args.panel_csv))}
    stop = reference_stop(args.train_csv, args.quantile) if args.add_stop else None

    records = read_fasta(args.in_fasta)
    counts: Counter = Counter()
    meta = {"label": args.label, "source": args.in_fasta,
            "stop_codon_appended": stop, "per_target": {}}
    with open(args.out, "w") as fh:
        for name, seq in records:
            target = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
            if target not in known:
                raise SystemExit(f"header {name!r} does not match a panel target {list(known)}")
            rna = seq.upper().replace("T", "U")
            cod = [rna[i:i + 3] for i in range(0, len(rna) - len(rna) % 3, 3)]
            added = False
            if stop and (not cod or cod[-1] not in STOP_CODONS):
                cod.append(stop)
                added = True
            body = cod[:-1] if cod and cod[-1] in STOP_CODONS else cod
            translated = "".join(GENETIC_CODE.get(c, "?") for c in body)
            prot = known[target]
            n = min(len(translated), len(prot))
            pct = 100.0 * sum(a == b for a, b in zip(translated, prot)) / max(len(prot), 1)
            i = counts[target]
            counts[target] += 1
            fh.write(f">{args.label}|{target}|{i} n_codon={len(cod)}\n{''.join(cod)}\n")
            meta["per_target"].setdefault(target, []).append(
                {"index": i, "n_codon": len(cod), "stop_appended": added,
                 "residue_identity_pct": pct, "exact_protein_match": translated == prot})
            print(f"{args.label}/{target}[{i}]: {len(cod)} codons, stop_appended={added}, "
                  f"residue identity {pct:.2f}%, exact match {translated == prot}")
    with open(args.out + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {sum(counts.values())} sequences -> {args.out}")


if __name__ == "__main__":
    main()
