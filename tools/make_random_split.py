"""Build a *random* 90/10 split over the same corpus, as the control arm.

The published protocol split randomly, which leaks homologs across the boundary
(measured: 79.8% of archaea val sequences have a >=50%-identity relative in train,
versus 0.52% after cluster splitting).  To quantify what that inflation is worth
in nats we need two models that differ *only* in the split rule -- same sequences,
same code, same hyperparameters -- so this writes a split file in exactly the
format ``scripts/04_write_codon_txt.py`` consumes.

Note this cannot be done by re-scoring the published checkpoints: those trained on
a random 90% of essentially this corpus, so they have already seen most of the
homology-clean validation set too.
"""
from __future__ import annotations

import argparse
import gzip
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="01's meta.tsv.gz")
    ap.add_argument("--out", required=True, help="split_random.tsv.gz")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--n-val", type=int, default=0,
                    help="exact validation count; use the cluster split's count so "
                         "the two arms differ only in the split rule")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    t0 = time.time()
    ids = []
    with gzip.open(args.meta, "rt") as fh:
        fh.readline()
        for line in fh:
            ids.append(line.split("\t", 1)[0])
    n = len(ids)

    rng = np.random.default_rng(args.seed)
    if args.n_val:
        is_val = np.zeros(n, dtype=bool)
        is_val[rng.choice(n, size=args.n_val, replace=False)] = True
    else:
        is_val = rng.random(n) < args.val_frac

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with gzip.open(args.out, "wt", compresslevel=6) as out:
        out.write("seq_id\tcluster_rep\tsplit\n")
        for sid, v in zip(ids, is_val):
            # each sequence is its own "cluster": that *is* the random protocol
            out.write(f"{sid}\t{sid}\t{'val' if v else 'train'}\n")

    n_val = int(is_val.sum())
    msg = (f"# make_random_split\n\n- meta: `{args.meta}`\n- seed: {args.seed}\n"
           f"- total sequences: {n:,}\n- train: {n - n_val:,} ({100*(n-n_val)/n:.2f}%)\n"
           f"- val: {n_val:,} ({100*n_val/n:.2f}%)\n"
           f"- output: `{args.out}` ({time.time()-t0:.0f}s)\n\n"
           "This is the **control arm**: the same corpus and the same code as "
           "`split_purified.tsv.gz`, the only difference being the split rule "
           "(random vs whole-cluster).\n")
    print(msg)
    if args.report:
        with open(args.report, "w") as fh:
            fh.write(msg)


if __name__ == "__main__":
    main()
