"""Build the SFT training set: top-quartile measured property, from the
homology-clean TRAIN split only (never val/test).

Property-agnostic: `--property expression` reads the fungal-expression split
(sft/data/fungal_expression_train.csv), `--property stability` the mRNA-stability
one, and the output prefix follows so the two never share a filename.

Unlike the paper's original three-step framework (train a predictor -> predict
over the full corpus -> filter by predicted score), these CSVs already carry a
REAL measured value for every sequence, so filtering uses ground truth directly.
No predictor is fit or needed for this step, which is what keeps the LightGBM
evaluator (evaluate/lightgbm_expression.py) genuinely independent of what built
the SFT set.

Output matches the LMDB format mrnagpt.data.CodonLMDBDataset already reads:
legacy-encoded entries [CLS][SEP] codons [SEP][SEP] (uint8), so no changes are
needed to the pretraining data-loading code -- the same 69->68 remap LUT
applies at read time.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys

import lmdb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sft.paths import legacy_root                                # noqa: E402

OLD_CLS, OLD_SEP = 2, 3
OLD_CODON0 = 5
OLD_CODONS = None  # filled below


def _legacy_vocab():
    global OLD_CODONS
    path = legacy_root() / "tokenizer/vocab.txt"
    toks = [l.strip() for l in open(path) if l.strip()]
    assert len(toks) == 69
    OLD_CODONS = toks[5:]
    return {c: i + OLD_CODON0 for i, c in enumerate(OLD_CODONS)}


def encode_legacy(codons: list[str], codon2id: dict) -> np.ndarray:
    ids = [OLD_CLS, OLD_SEP] + [codon2id[c] for c in codons] + [OLD_SEP, OLD_SEP]
    return np.array(ids, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", default="expression",
                    help="which measured property: names the output prefix, and picks the "
                         "default --train-csv/--clusters for the known datasets "
                         "(expression -> fungal_expression, stability -> mrna_stability)")
    ap.add_argument("--train-csv", default=None)
    ap.add_argument("--quantile", type=float, default=0.75,
                    help="keep sequences with measured Value above this quantile of train")
    ap.add_argument("--low-threshold", type=float, default=None,
                    help="absolute cut for the LOW control (Value <= this) instead of "
                         "size-matching it to the high arm. Needed when the value "
                         "distribution is skewed: size-matching a large high arm can "
                         "force the control to reach up into mid/high-value sequences, "
                         "which destroys the control.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="absolute cut on the measured Value instead of a quantile "
                         "(e.g. --threshold 2 for translation efficiency > 2). The "
                         "low-direction control is then size-matched to the high set "
                         "by taking that many lowest-Value sequences, so the two arms "
                         "still differ only in which end of the measurement they saw.")
    ap.add_argument("--direction", choices=["high", "low"], default="high",
                    help="high: top (1-quantile) of the measured property -- the SFT set. "
                         "low: the mirror-image bottom quantile, used as a NEGATIVE CONTROL. "
                         "Fine-tuning on it with identical hyperparameters tests whether the "
                         "direction generated sequences move in is set by the measured property "
                         "or is just an artifact of fine-tuning on any small subset of this host.")
    ap.add_argument("--prefix", default=None,
                    help="output basename prefix (defaults to sft_high/low_<property>)")
    ap.add_argument("--out-dir", default="sft/data")
    ap.add_argument("--clusters", default=None)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    DATASETS = {"expression": ("fungal_expression_train.csv", "fungal_clusters.tsv.gz"),
                "stability": ("mrna_stability_train.csv", "mrna_stability_clusters.tsv.gz"),
                "translation_efficiency": ("ecoli_te_train.csv", "ecoli_te_clusters.tsv.gz"),
                "bacteria_expression": ("bacteria_expression_train.csv",
                                        "bacteria_expression_clusters.tsv.gz")}
    if args.train_csv is None or args.clusters is None:
        if args.property not in DATASETS:
            raise SystemExit(f"--property {args.property} is not a known dataset "
                             f"({sorted(DATASETS)}); pass --train-csv and --clusters "
                             "explicitly")
        csv_name, clu_name = DATASETS[args.property]
        args.train_csv = args.train_csv or os.path.join(args.out_dir, csv_name)
        args.clusters = args.clusters or os.path.join(args.out_dir, clu_name)

    rows = list(csv.DictReader(open(args.train_csv)))
    vals = sorted(float(r["Value"]) for r in rows)
    if args.threshold is not None:
        if args.direction == "high":
            threshold, rel = args.threshold, ">="
            kept = {r["seq_id"]: r for r in rows if float(r["Value"]) >= threshold}
        elif args.low_threshold is not None:
            threshold, rel = args.low_threshold, "<="
            kept = {r["seq_id"]: r for r in rows if float(r["Value"]) <= threshold}
        else:
            # size-matched mirror: the same number of sequences, from the bottom
            n_high = sum(1 for v in vals if v >= args.threshold)
            threshold, rel = vals[max(n_high - 1, 0)], "<="
            order = sorted(rows, key=lambda r: float(r["Value"]))[:n_high]
            kept = {r["seq_id"]: r for r in order}
        pct = None
    elif args.direction == "high":
        threshold = vals[int(len(vals) * args.quantile)]
        kept = {r["seq_id"]: r for r in rows if float(r["Value"]) >= threshold}
        pct = int(args.quantile * 100)
        rel = ">="
    else:
        threshold = vals[int(len(vals) * (1.0 - args.quantile))]
        kept = {r["seq_id"]: r for r in rows if float(r["Value"]) <= threshold}
        pct = int((1.0 - args.quantile) * 100)
        rel = "<="
    prefix = args.prefix or f"sft_{args.direction}_{args.property}"
    cut = f"P{pct} ({threshold:.4f})" if pct is not None else f"{threshold:.4f} (absolute)"
    print(f"train: {len(rows)} total, {args.direction} subset = Value {rel} {cut}, "
          f"kept {len(kept)} ({100*len(kept)/len(rows):.1f}%)")

    # covariate check: low-expression genes may differ in length/GC for reasons that
    # have nothing to do with codon choice, which would confound the control
    def _stats(sel):
        L = [len(r["Sequence"]) for r in sel]
        G = [sum(c in "GC" for c in r["Sequence"].upper()) / max(len(r["Sequence"]), 1) for r in sel]
        return (sum(L) / len(L), sorted(L)[len(L) // 2], 100 * sum(G) / len(G))
    other = [r for r in rows if r["seq_id"] not in kept]
    for nm, sel in (("kept", list(kept.values())), ("rest", other)):
        m, md, gc = _stats(sel)
        print(f"  [{nm}] n={len(sel)} length mean {m:.0f} median {md} nt, GC {gc:.2f}%")

    # Sub-split the filtered set into SFT-train/SFT-val by whole cluster, reusing
    # the clustering already computed over the full corpus (reports/*_leakage.md)
    # rather than re-clustering.
    import gzip as _gzip
    member2rep = {}
    with _gzip.open(args.clusters, "rt") as f:
        for line in f:
            rep, mem = line.strip().split("\t")
            member2rep[mem] = rep
    clusters_present = {}
    for sid in kept:
        clusters_present.setdefault(member2rep[sid], []).append(sid)

    rng = __import__("random").Random(args.seed)
    reps = list(clusters_present)
    rng.shuffle(reps)
    n_val_target = max(1, int(len(kept) * args.val_frac))
    val_ids, train_ids = [], []
    for rep in reps:
        (val_ids if len(val_ids) < n_val_target else train_ids).extend(clusters_present[rep])
    print(f"SFT split (whole-cluster, reused from the full-corpus clustering): "
          f"train={len(train_ids)} val={len(val_ids)}")

    codon2id = _legacy_vocab()

    def write_split(ids, label):
        txt_path = os.path.join(args.out_dir, f"{prefix}_{label}_codon.txt.gz")
        lmdb_path = os.path.join(args.out_dir, f"{prefix}_{label}.lmdb")
        entries = []
        with gzip.open(txt_path, "wt") as f:
            for sid in ids:
                seq = kept[sid]["Sequence"].upper()
                codons = [seq[i:i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]
                if any(c not in codon2id for c in codons):
                    continue
                f.write(" ".join(codons) + "\n")
                entries.append(codons)
        env = lmdb.open(lmdb_path, subdir=False, map_size=2 * 10**9)
        with env.begin(write=True) as txn:
            for i, codons in enumerate(entries):
                txn.put(str(i).encode(), encode_legacy(codons, codon2id).tobytes())
        env.close()
        from mrnagpt.data import build_lengths
        build_lengths(lmdb_path)
        print(f"{label}: wrote {len(entries)} entries -> {lmdb_path}")

    write_split(train_ids, "train")
    write_split(val_ids, "val")


if __name__ == "__main__":
    main()
