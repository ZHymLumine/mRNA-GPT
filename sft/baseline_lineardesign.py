"""Baseline 2: LinearDesign (Zhang et al., Nature 2023).

Runs the compiled binary directly (bin/LinearDesign_2D <lambda> <verbose>
<codon_usage_csv>, protein on stdin) because the shipped `lineardesign` wrapper
is a python2 script and this cluster has no python2 -- the wrapper does nothing
but forward those three arguments.

lambda trades folding energy against CAI: 0.0 is pure MFE, larger values pull
toward the codon-usage table.  Sweeping it is the honest way to compare against
a model that was not given such a knob.

Two codon-usage tables are run:
  - the yeast table shipped with LinearDesign (its own fungal-relevant default)
  - a table built from OUR top-quartile fungal reference set, so LinearDesign
    gets exactly the codon-preference information the SFT model was trained on

LinearDesign emits no terminal stop codon, so one is appended (the most frequent
stop in the same reference set) -- otherwise every sequence would count as an
invalid CDS for a reason that has nothing to do with the method.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrnagpt.vocab import GENETIC_CODE, STOP_CODONS
from sft.generate_target_panel import load_targets
from sft.paths import external

# LinearDesign is third-party software and is not included in this repository:
# install it yourself (https://github.com/LinearDesignSoftware/LinearDesign),
# build bin/LinearDesign_2D, and set LINEARDESIGN_DIR to the checkout -- or put
# it at <MRNA_GPT_EXTERNAL>/LinearDesign.
LD_DIR = str(external("LinearDesign", "LINEARDESIGN_DIR"))


def to_codons(seq: str) -> list[str]:
    seq = seq.upper().replace("T", "U")
    return [seq[i:i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def build_usage_table(train_csv: str, quantile: float, out_path: str,
                      threshold: float | None = None) -> str:
    """Write LinearDesign's codon-usage CSV format: codon,aa,within-family frequency."""
    rows = list(csv.DictReader(open(train_csv)))
    vals = sorted(float(r["Value"]) for r in rows)
    thr = threshold if threshold is not None else vals[int(len(vals) * quantile)]
    counts: Counter = Counter()
    for r in rows:
        if float(r["Value"]) >= thr:
            counts.update(to_codons(r["Sequence"]))
    # Every one of the 64 codons must appear: LinearDesign rejects a table with
    # fewer ("Codon frequency file needs to contain 64 codons!"). Emitting only
    # OBSERVED codons breaks on any reference set that never uses one -- the
    # bacterial expression library ends every construct in TGA, so TAA and TAG
    # are absent and the table came out with 62 rows.
    by_aa: dict[str, dict[str, int]] = defaultdict(dict)
    for c, aa in GENETIC_CODE.items():
        by_aa[aa][c] = 0
    for c, n in counts.items():
        aa = GENETIC_CODE.get(c)
        if aa:
            by_aa[aa][c] = n
    n_zero = sum(1 for aa in by_aa for c in by_aa[aa] if by_aa[aa][c] == 0)
    if n_zero:
        print(f"  note: {n_zero} codon(s) never used in the reference set, written "
              f"with frequency 0: "
              f"{sorted(c for aa in by_aa for c in by_aa[aa] if by_aa[aa][c] == 0)}")
    with open(out_path, "w") as fh:
        fh.write("#,,\n")
        for aa in sorted(by_aa):
            tot = sum(by_aa[aa].values())
            for c in sorted(by_aa[aa]):
                freq = max(by_aa[aa][c] / tot, 1e-4) if tot else 1e-4
                fh.write(f"{c},{aa},{freq:.4f}\n")
    stops = {c: n for c, n in counts.items() if c in STOP_CODONS}
    return max(stops, key=stops.get) if stops else "UAA"


def run_lineardesign(protein: str, lam: float, table: str) -> dict:
    proc = subprocess.run([os.path.join(LD_DIR, "bin/LinearDesign_2D"), str(lam), "0", table],
                          input=protein + "\n", capture_output=True, text=True,
                          cwd=LD_DIR, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"LinearDesign failed (lambda={lam}): {proc.stderr[-500:]}")
    seq = mfe = cai = None
    for line in proc.stdout.splitlines():
        # the binary interleaves "j=<i>" progress markers with the result lines
        if "mRNA sequence:" in line:
            seq = line.split("mRNA sequence:")[1].strip()
        elif "folding free energy" in line:
            head, _, tail = line.partition(";")
            mfe = float(head.split(":")[1].strip().split()[0])
            cai = float(tail.split(":")[1].strip())
    if seq is None:
        raise RuntimeError(f"could not parse LinearDesign output: {proc.stdout[-500:]}")
    return {"seq": seq, "ld_mfe": mfe, "ld_cai": cai}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/protein_sequences.csv")
    ap.add_argument("--train-csv", default="sft/data/fungal_expression_train.csv")
    ap.add_argument("--quantile", type=float, default=0.75)
    ap.add_argument("--threshold", type=float, default=None,
                    help="absolute cut on Value instead of a quantile (e.g. 2 for TE), "
                         "mirroring sft/prepare_sft_data.py")
    ap.add_argument("--lambdas", default="0,1,4")
    ap.add_argument("--tables", default="ours,yeast",
                    help="ours = built here from --train-csv's top quantile; yeast/human = "
                         "shipped with LinearDesign. 'fungal' is accepted as a legacy alias "
                         "of 'ours'.")
    ap.add_argument("--table-label", default="fungal",
                    help="name for the 'ours' table in output labels and filenames "
                         "(e.g. stability -> lineardesign_stability_l0)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--table-path", default=None,
                    help="reuse a prebuilt fungal usage table instead of writing one "
                         "(lets several lambdas run in parallel without racing on the file)")
    ap.add_argument("--meta-name", default="lineardesign_meta.json")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ours_table = args.table_path or os.path.join(
        args.out_dir, f"codon_usage_{args.table_label}_p{int(args.quantile*100)}.csv")
    if args.table_path:
        # the stop codon is a one-line recount; the table itself is reused as-is
        rows = list(csv.DictReader(open(args.train_csv)))
        vals = sorted(float(r["Value"]) for r in rows)
        thr = args.threshold if args.threshold is not None \
            else vals[int(len(vals) * args.quantile)]
        counts: Counter = Counter()
        for r in rows:
            if float(r["Value"]) >= thr:
                counts.update(to_codons(r["Sequence"]))
        stops = {c: n for c, n in counts.items() if c in STOP_CODONS}
        stop_codon = max(stops, key=stops.get) if stops else "UAA"
    else:
        stop_codon = build_usage_table(args.train_csv, args.quantile, ours_table,
                                       args.threshold)
    tables = {"ours": ours_table, "fungal": ours_table,   # 'fungal' kept as a legacy alias
              "yeast": os.path.join(LD_DIR, "codon_usage_freq_table_yeast.csv"),
              "human": os.path.join(LD_DIR, "codon_usage_freq_table_human.csv")}
    label_of = {"ours": args.table_label, "fungal": args.table_label,
                "yeast": "yeast", "human": "human"}

    targets = load_targets(args.csv)
    meta: dict = {"stop_codon_appended": stop_codon, "lambdas": args.lambdas, "runs": {}}
    for tname in args.tables.split(","):
        for lam in [float(x) for x in args.lambdas.split(",")]:
            label = f"lineardesign_{label_of[tname]}_l{lam:g}"
            path = os.path.join(args.out_dir, f"{label}.fasta")
            recs = {}
            with open(path, "w") as fh:
                for target, prot in targets:
                    r = run_lineardesign(prot, lam, tables[tname])
                    codons = to_codons(r["seq"])
                    if not codons or codons[-1] not in STOP_CODONS:
                        codons.append(stop_codon)
                    fh.write(f">{label}|{target}|0 n_codon={len(codons)}\n{''.join(codons)}\n")
                    recs[target] = {"ld_mfe": r["ld_mfe"], "ld_cai": r["ld_cai"],
                                    "n_codon": len(codons)}
                    print(f"{label} {target}: LD-reported MFE {r['ld_mfe']} CAI {r['ld_cai']}",
                          flush=True)
            meta["runs"][label] = {"fasta": path, "per_target": recs}
    with open(os.path.join(args.out_dir, args.meta_name), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {len(meta['runs'])} LinearDesign variants -> {args.out_dir}")


if __name__ == "__main__":
    main()
