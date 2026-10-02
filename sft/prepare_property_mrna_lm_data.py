"""Build a measured-property dataset in mRNA-LM's input format, for training a
host-matched NEURAL evaluator.

Why this exists: the only tree-based predictor we have (LightGBM) takes CAI as an
input feature -- 19.6% of its total gain on the fungal task, r=0.88 with CAI over
generated sequences -- so ranking CAI-maximizing methods by it is near
tautological. A network reading the codon sequence directly has no such
hand-built shortcut.

Two protocols, selected with --protocol:

  standard   (default) TRAIN split -> training, VAL -> model selection,
             TEST -> the reported held-out performance. This is the ordinary
             supervised protocol and uses the largest split, so it produces the
             stronger evaluator. It is not circular in the sense one might
             worry about: the SFT training set is selected on MEASURED values, with no
             predictor in the loop, so this predictor never influenced what the
             generative model was trained on. It does see, with labels, the
             sequences the model was fine-tuned to imitate, so it must be read
             alongside the memorisation check (nearest-neighbour identity to the
             fine-tuning set) and the predictor-free metrics.

  valtest    trains on VAL+TEST only, leaving the TRAIN split -- and therefore
             the SFT subset and all of its homologs -- entirely unseen. A weaker
             evaluator (less data) but with no shared sequences at all.

Either way the splits are whole-cluster homology-clean at 50% nucleotide
identity, so no fold boundary is crossed by a homolog.

⚠️ DO NOT substitute mRNA-LM's own bundled half-life head for the stability
task. Its `data/mrna_half-life.csv` shares **76.9% of its CDSs exactly** with
mRNA_Stability.csv -- both descend from the same human half-life compendium --
so 30.6% of our SFT-high training subset sits inside its supervised training
data. Measured, not assumed: see the overlap check in the README. (Its loader
`build_saluki_dataset` in dataload.py is also broken: `load_dataset` takes one
argument and is called with two.) The fungal precedent -- LoRA on our own
VAL+TEST -- is the only clean route here.

Neither dataset carries UTRs, so one fixed UTR context is used for every record,
the same context the designs are scored under. The 5'/3' encoders therefore see
constant input and the CDS branch carries the signal; this is a CDS-only
predictor wearing mRNA-LM's architecture, and it must be described that way.
The context is host-matched:

    expression / fungal   -> yeast ADH1
    stability  / human    -> a human transcript from mRNA-LM's own test fold
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sft.paths import external                                   # noqa: E402

DATASETS = {
    # property   : (basename,             clusters,                        default context)
    "expression": ("fungal_expression", "fungal_clusters.tsv.gz", "adh1_yeast"),
    "stability": ("mrna_stability", "mrna_stability_clusters.tsv.gz", "human_median"),
    "translation_efficiency": ("ecoli_te", "ecoli_te_clusters.tsv.gz", "ecoli_rbs"),
    "bacteria_expression": ("bacteria_expression", "bacteria_expression_clusters.tsv.gz",
                            "ecoli_rbs"),
}

# A single fixed prokaryotic context for the E. coli TE dataset, which carries no
# UTRs. Canonical Shine-Dalgarno (AGGAGG) with the standard 7 nt spacer, plus a
# rho-independent terminator-like 3' segment. The identity of this context cannot
# affect any ranking: it is IDENTICAL for every record and every scored design, so
# the 5'/3' encoders see constant input and the CDS branch carries all the signal.
# Report the resulting model as a CDS-only predictor in mRNA-LM's architecture.
ECOLI_5UTR = ("GGGAATTGTGAGCGGATAACAATTCCCCTCTAGAAATAATTTTGTTTAACTTTAAGAAGGAGATATACAT")
ECOLI_3UTR = ("CTGTTGAACAACTGAACTAGCATAACCCCTTGGGGCCTCTAAACGGGTCTTGAGGGGTTTTTTG")
# mRNA-LM is third-party software and is not included in this repository: clone
# it yourself (https://github.com/sunjinyuan/mRNA-LM) and set MRNA_LM_REPO to the
# checkout -- or put it at <MRNA_GPT_EXTERNAL>/mRNA-LM.
MRNA_LM = str(external("mRNA-LM", "MRNA_LM_REPO"))


def load_clusters(path: str) -> dict[str, str]:
    member2rep = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            rep, mem = line.strip().split("\t")
            member2rep[mem] = rep
    return member2rep


def human_median_utr(tr_csv: str, fold: str = "5") -> tuple[str, str, str]:
    """The transcript in mRNA-LM's own held-out fold whose UTR lengths are
    closest to that fold's medians. Same rule as
    sft/score_panel_translation_rate.py:pick_human_utr, reimplemented on the csv
    module so this script does not need pandas."""
    csv.field_size_limit(10 ** 9)
    rows = [r for r in csv.DictReader(open(tr_csv)) if str(r.get("split")) == fold]
    if not rows:
        raise SystemExit(f"no rows with split={fold} in {tr_csv}")
    m5 = statistics.median(len(r["UTR5"] or "") for r in rows)
    m3 = statistics.median(len(r["UTR3"] or "") for r in rows)
    best = min(rows, key=lambda r: abs(len(r["UTR5"] or "") - m5) / max(m5, 1)
               + abs(len(r["UTR3"] or "") - m3) / max(m3, 1))
    return best["UTR5"] or "", best["UTR3"] or "", best.get("ENSTID", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", default="expression", choices=sorted(DATASETS))
    ap.add_argument("--data-dir", default="sft/data")
    ap.add_argument("--protocol", default="standard", choices=["standard", "valtest"],
                    help="standard: TRAIN->train, VAL->select, TEST->report. "
                         "valtest: train on VAL+TEST only, never touching TRAIN.")
    ap.add_argument("--train-csv", default=None)
    ap.add_argument("--val-csv", default=None)
    ap.add_argument("--test-csv", default=None)
    ap.add_argument("--clusters", default=None)
    ap.add_argument("--utr-context", default=None,
                    choices=["adh1_yeast", "human_median", "ecoli_rbs"],
                    help="fixed UTR pair wrapped around every CDS; must match what "
                         "sft/score_panel_translation_rate.py --contexts uses at scoring "
                         "time, or the evaluator is off-distribution")
    ap.add_argument("--tr-csv", default=os.path.join(MRNA_LM, "data/translation_rate.csv"),
                    help="source of the human UTR context")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    name, clu, default_ctx = DATASETS[args.property]
    train_csv = args.train_csv or os.path.join(args.data_dir, f"{name}_train.csv")
    val_csv = args.val_csv or os.path.join(args.data_dir, f"{name}_val.csv")
    test_csv = args.test_csv or os.path.join(args.data_dir, f"{name}_test.csv")
    clusters = args.clusters or os.path.join(args.data_dir, clu)
    ctx_name = args.utr_context or default_ctx
    suffix = "" if args.protocol == "standard" else "_valtest"
    out = args.out or os.path.join(MRNA_LM, "data", f"{name}_lm{suffix}.csv")

    if ctx_name == "adh1_yeast":
        from sft.data.adh1_utr_context import ADH1_3UTR, ADH1_5UTR
        u5, u3, ctx_id = ADH1_5UTR, ADH1_3UTR, "yeast ADH1"
    elif ctx_name == "ecoli_rbs":
        u5, u3, ctx_id = ECOLI_5UTR, ECOLI_3UTR, "E. coli consensus SD / terminator"
    else:
        u5, u3, ctx_id = human_median_utr(args.tr_csv)
    print(f"UTR context: {ctx_name} ({ctx_id}) -- 5' {len(u5)} nt / 3' {len(u3)} nt")

    # fold 1-3 -> training, 4 -> model selection, 5 -> held-out test. The
    # downstream trainer reads exactly that convention.
    rows = []
    if args.protocol == "standard":
        # reuse the dataset's own homology-clean splits directly: no resampling,
        # so the reported test number is on the same TEST split every other
        # evaluator is reported on.
        member2rep = load_clusters(clusters)
        train_rows = list(csv.DictReader(open(train_csv)))
        by_cluster: dict[str, list] = {}
        for r in train_rows:
            by_cluster.setdefault(member2rep.get(r["seq_id"], r["seq_id"]), []).append(r)
        reps = sorted(by_cluster)
        random.Random(args.seed).shuffle(reps)
        for i, rep in enumerate(reps):                 # spread TRAIN over folds 1-3
            for r in by_cluster[rep]:
                rows.append((r, (i % 3) + 1))
        for r in csv.DictReader(open(val_csv)):
            rows.append((r, 4))
        for r in csv.DictReader(open(test_csv)):
            rows.append((r, 5))
        print(f"standard protocol: TRAIN {len(train_rows)} -> folds 1-3, "
              f"VAL -> fold 4, TEST -> fold 5 "
              f"({len(reps)} whole clusters within TRAIN)")
    else:
        heldout = []
        for path in (val_csv, test_csv):
            heldout.extend(list(csv.DictReader(open(path))))
        member2rep = load_clusters(clusters)
        by_cluster = {}
        for r in heldout:
            by_cluster.setdefault(member2rep.get(r["seq_id"], r["seq_id"]), []).append(r)
        reps = sorted(by_cluster)
        random.Random(args.seed).shuffle(reps)
        for i, rep in enumerate(reps):
            for r in by_cluster[rep]:
                rows.append((r, (i % args.folds) + 1))
        print(f"valtest protocol: {len(heldout)} sequences from VAL+TEST only "
              f"({len(reps)} clusters -> {args.folds} whole-cluster folds)")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    counts: dict[int, int] = {}
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["UTR5", "CDS", "UTR3", "y", "split", "seq_id"])
        for r, f in rows:
            w.writerow([u5, r["Sequence"].upper(), u3, r["Value"], f, r["seq_id"]])
            counts[f] = counts.get(f, 0) + 1
    print(f"fold sizes: {dict(sorted(counts.items()))}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
