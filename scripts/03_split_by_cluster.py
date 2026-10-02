#!/usr/bin/env python3
"""
03 - 90/10 train/val split along MMseqs2 clusters (sequences from one cluster
never straddle train and val).

The original pipeline shuffled every 10,000 lines with random.shuffle in
save_lmdb.py and cut at 0.9, which put homologous gene families on both the
train and the val side.
"""
import argparse
import gzip
import os
import random
import time

import numpy as np


def read_clusters(path):
    """Return (member_ids: list[str], cluster_of: np.int32[n], n_clusters)."""
    rep_to_cid = {}
    members = []
    cluster_of = []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            rep, mem = line.rstrip("\n").split("\t")
            cid = rep_to_cid.get(rep)
            if cid is None:
                cid = len(rep_to_cid)
                rep_to_cid[rep] = cid
            members.append(mem)
            cluster_of.append(cid)
    reps = [None] * len(rep_to_cid)
    for rep, cid in rep_to_cid.items():
        reps[cid] = rep
    return members, np.asarray(cluster_of, dtype=np.int32), reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--meta", required=True, help="meta.tsv.gz produced by 01, used for taxonomy statistics")
    ap.add_argument("--out", required=True, help="split.tsv.gz")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overshoot", type=float, default=0.05,
                    help="how far val may overshoot the target fraction (keeps a giant cluster from blowing past it)")
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    t0 = time.time()
    members, cluster_of, reps = read_clusters(args.clusters)
    n_seq = len(members)
    n_clu = len(reps)
    print(f"[{args.domain}] {n_seq} sequences in {n_clu} clusters ({time.time()-t0:.0f}s)",
          flush=True)

    sizes = np.bincount(cluster_of, minlength=n_clu)

    # ---- shuffle clusters, greedily fill val until val_frac is reached ----
    # Simply accumulating in random order until target is hit does not work:
    # connected-component clustering can produce giant clusters, and one of them
    # arriving early would push val far beyond 10% in a single step.  Clusters
    # that would overshoot are skipped instead.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_clu)
    target = int(round(args.val_frac * n_seq))
    cap = int(target * (1 + args.overshoot))

    is_val_cluster = np.zeros(n_clu, dtype=bool)
    acc = 0
    n_skipped_big = 0
    for cid in order:
        if acc >= target:
            break
        s = int(sizes[cid])
        if acc + s > cap:
            n_skipped_big += 1
            continue
        is_val_cluster[cid] = True
        acc += s
    n_val_clu = int(is_val_cluster.sum())
    if acc < target * 0.95:
        print(f"[{args.domain}] warning: val only filled to {acc}/{target}, "
              f"probably too many giant clusters (largest cluster {int(sizes.max()):,} seqs = "
              f"{sizes.max()/n_seq:.2%})", flush=True)
    is_val = is_val_cluster[cluster_of]
    n_val = int(is_val.sum())
    n_train = n_seq - n_val
    print(f"[{args.domain}] val: {n_val_clu} clusters / {n_val} seqs "
          f"({n_val/n_seq:.4%}); train: {n_clu-n_val_clu} clusters / {n_train} seqs",
          flush=True)

    # ---- assertion: no cluster straddles the split ----
    assert not (is_val_cluster[cluster_of[~is_val]]).any(), "a val cluster leaked into train"
    assert (is_val_cluster[cluster_of[is_val]]).all(), "a train cluster leaked into val"

    # ---- write split.tsv.gz ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with gzip.open(args.out, "wt", compresslevel=6) as out:
        out.write("seq_id\tcluster_rep\tsplit\n")
        for i, mem in enumerate(members):
            out.write(f"{mem}\t{reps[cluster_of[i]]}\t{'val' if is_val[i] else 'train'}\n")
    print(f"[{args.domain}] wrote {args.out} ({time.time()-t0:.0f}s)", flush=True)

    # ---- assertion: seq_ids match meta exactly ----
    meta_ids = set()
    phy_of = {}
    with gzip.open(args.meta, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        i_id, i_phy = header.index("seq_id"), header.index("phylum")
        for line in fh:
            p = line.rstrip("\n").split("\t")
            meta_ids.add(p[i_id])
            phy_of[p[i_id]] = p[i_phy]
    mem_set = set(members)
    assert mem_set == meta_ids, (
        f"seq_ids in clusters and meta disagree: "
        f"only_in_clusters={len(mem_set-meta_ids)}, only_in_meta={len(meta_ids-mem_set)}")

    phy_train, phy_val = set(), set()
    for i, mem in enumerate(members):
        (phy_val if is_val[i] else phy_train).add(phy_of.get(mem, ""))
    phy_train.discard(""); phy_val.discard("")

    # ---- report ----
    nz = sizes[sizes > 0]
    with open(args.report, "w") as r:
        r.write(f"# 03_split_by_cluster - {args.domain}\n\n")
        r.write(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        r.write(f"- cluster file: `{args.clusters}`\n")
        r.write(f"- split seed: {args.seed}, target val fraction: {args.val_frac:.0%}\n\n")
        r.write("## Cluster structure\n\n")
        r.write(f"- total sequences: {n_seq:,}\n")
        r.write(f"- total clusters: {n_clu:,} (compression ratio {n_seq/n_clu:.2f}x)\n")
        r.write(f"- singleton clusters: {int((nz==1).sum()):,} ({(nz==1).mean():.2%} of clusters, "
                f"covering {int(nz[nz==1].sum())/n_seq:.2%} of sequences)\n")
        r.write(f"- sequences in multi-member clusters: {n_seq-int(nz[nz==1].sum()):,} "
                f"({1-int(nz[nz==1].sum())/n_seq:.2%})\n")
        r.write(f"- cluster size P50/P90/P99/max: {int(np.percentile(nz,50))} / "
                f"{int(np.percentile(nz,90))} / {int(np.percentile(nz,99))} / {int(nz.max())}\n")
        r.write(f"- largest cluster as a share of all sequences: {nz.max()/n_seq:.2%}"
                f" (clusters skipped for exceeding the val quota: {n_skipped_big:,})\n\n")
        r.write("## Split result\n\n")
        r.write("| | clusters | sequences | share of sequences |\n|---|---:|---:|---:|\n")
        r.write(f"| train | {n_clu-n_val_clu:,} | {n_train:,} | {n_train/n_seq:.2%} |\n")
        r.write(f"| val | {n_val_clu:,} | {n_val:,} | {n_val/n_seq:.2%} |\n\n")
        r.write(f"- phyla covered by train: {len(phy_train)}\n")
        r.write(f"- phyla covered by val: {len(phy_val)}\n")
        r.write(f"- phyla unique to val: {sorted(phy_val - phy_train) or 'none'}\n")
        r.write(f"- phyla unique to train: {sorted(phy_train - phy_val) or 'none'}\n\n")
        r.write("**Assertions passed**: no cluster appears in both train and val; "
                "the seq_id set of clusters.tsv matches meta.tsv.gz exactly.\n")
    print(open(args.report).read())


if __name__ == "__main__":
    main()
