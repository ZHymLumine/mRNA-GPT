#!/usr/bin/env python3
"""
09 - Validation purification: drive the residual homology to zero

Background: the connected-component cluster split brings >=50% homology leakage
down from 79.8% to ~15% (measured on archaea); what remains are distant
homologous pairs that cascaded clustering did not connect.  This step finds and
removes them with an exhaustive sensitive search.

Two things must be done at cluster granularity (otherwise new leakage is
introduced):

  1. **Capping moves whole clusters**: when val exceeds the cap, moving only part
     of a cluster's members into train would leave their cluster-mates in val
     >=50% similar to them, creating leakage instead.  So clusters are kept or
     dropped as a whole.

  2. **Removal moves whole clusters**: moving only the one matching val sequence
     into train would immediately turn its cluster-mates in val into a new source
     of leakage.  So as soon as any member matches, the whole cluster is moved.

The search target is **all sequences** (train union val), not just train: val
sequences that get moved become part of train, and leaving them out of the target
would require repeated iteration.  Searching against everything and counting only
*cross-cluster* hits as leakage converges in a single pass -- a val cluster that
survives has no >=50% hit against **any** other cluster, and therefore none
against train either.
"""
import argparse
import gzip
import os
import subprocess
import time
from collections import defaultdict

import numpy as np


def run(cmd, **kw):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def read_split(path):
    """seq_id -> (cluster_rep, split), preserving the original order."""
    ids, reps, splits = [], [], []
    with gzip.open(path, "rt") as fh:
        fh.readline()
        for line in fh:
            sid, rep, sp = line.rstrip("\n").split("\t")
            ids.append(sid); reps.append(rep); splits.append(sp)
    return ids, reps, splits


def write_subset(cds_gz, wanted, out_fa):
    n = 0
    p = subprocess.Popen(["unpigz", "-c", cds_gz], stdout=subprocess.PIPE, bufsize=1 << 22)
    with open(out_fa, "wb") as out:
        keep = False
        for line in p.stdout:
            if line[:1] == b">":
                keep = line[1:].rstrip().decode() in wanted
                if keep:
                    n += 1
            if keep:
                out.write(line)
    p.stdout.close(); p.wait()
    assert n == len(wanted), f"{out_fa}: extracted {n} sequences, expected {len(wanted)}"
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cds", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True, help="the purified split.tsv.gz")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--max-val", type=int, default=500_000,
                    help="cap on the val sequences sent to the search (applied whole clusters at a time), to bound search cost")
    ap.add_argument("--min-seq-id", default="0.5")
    ap.add_argument("--cov", default="0.8")
    ap.add_argument("--sens", default="7.5")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--split-memory-limit", default="0",
                    help="per-split memory cap for the mmseqs prefilter (e.g. 200G). "
                         "0 = use all available memory, which gets OOM-killed inside a "
                         "PBS cgroup, so set it below your quota")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.work, exist_ok=True)
    ids, reps, splits = read_split(args.split)
    n_seq = len(ids)

    members_of = defaultdict(list)
    for sid, rep in zip(ids, reps):
        members_of[rep].append(sid)
    val_clusters = sorted({r for r, s in zip(reps, splits) if s == "val"})
    n_val0 = sum(1 for s in splits if s == "val")
    print(f"[{args.domain}] input: {n_seq:,} sequences, val {n_val0:,} sequences / "
          f"{len(val_clusters):,} clusters", flush=True)

    # ---- 1. cap val, whole clusters at a time ----
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(val_clusters))
    kept_clusters, acc, n_capped_clu = set(), 0, 0
    for i in order:
        c = val_clusters[i]
        s = len(members_of[c])
        if acc + s > args.max_val:
            n_capped_clu += 1
            continue
        kept_clusters.add(c)
        acc += s
    n_val_capped = acc
    print(f"[{args.domain}] val after capping: {n_val_capped:,} sequences / "
          f"{len(kept_clusters):,} clusters (clusters returned to train for exceeding "
          f"the cap: {n_capped_clu:,})", flush=True)

    # ---- 2. exhaustive search: candidate val against everything ----
    q_ids = {sid for c in kept_clusters for sid in members_of[c]}
    q_fa, t_fa = f"{args.work}/q.fa", f"{args.work}/all.fa"
    write_subset(args.cds, q_ids, q_fa)
    if not os.path.exists(t_fa):
        run(["bash", "-c", f"unpigz -c {args.cds} > {t_fa}"])

    tsv = f"{args.work}/hits.tsv"
    if not os.path.exists(tsv):
        run(["mmseqs", "createdb", q_fa, f"{args.work}/qDB", "--dbtype", "2"],
            stdout=subprocess.DEVNULL)
        run(["mmseqs", "createdb", t_fa, f"{args.work}/tDB", "--dbtype", "2"],
            stdout=subprocess.DEVNULL)
        run(["mmseqs", "search", f"{args.work}/qDB", f"{args.work}/tDB",
             f"{args.work}/res", f"{args.work}/tmp",
             "--search-type", "3", "--strand", "1", "-s", args.sens,
             "--min-seq-id", args.min_seq_id, "-c", args.cov, "--cov-mode", "0",
             "-e", "1e-3", "--max-seqs", "300",
             "--split-memory-limit", args.split_memory_limit,
             "--threads", str(args.threads), "--remove-tmp-files", "1"],
            stdout=subprocess.DEVNULL)
        run(["mmseqs", "convertalis", f"{args.work}/qDB", f"{args.work}/tDB",
             f"{args.work}/res", tsv, "--format-output", "query,target,pident",
             "--threads", str(args.threads)], stdout=subprocess.DEVNULL)
    print(f"[{args.domain}] search finished ({time.time()-t0:.0f}s)", flush=True)

    # ---- 3. if any member has a cross-cluster hit, move the whole cluster to train ----
    rep_of = dict(zip(ids, reps))
    condemned = set()
    n_hits = 0
    with open(tsv) as fh:
        for line in fh:
            q, t, _pid = line.rstrip("\n").split("\t")
            rq = rep_of[q]
            if rep_of[t] == rq:
                continue          # a same-cluster hit is expected and is not leakage
            n_hits += 1
            condemned.add(rq)
    final_val_clusters = kept_clusters - condemned
    n_val_final = sum(len(members_of[c]) for c in final_val_clusters)
    print(f"[{args.domain}] {n_hits:,} cross-cluster hits involving {len(condemned):,} "
          f"val clusters -> whole clusters moved back to train", flush=True)
    print(f"[{args.domain}] val after purification: {n_val_final:,} sequences / "
          f"{len(final_val_clusters):,} clusters", flush=True)

    # ---- 4. write out ----
    with gzip.open(args.out, "wt", compresslevel=6) as out:
        out.write("seq_id\tcluster_rep\tsplit\n")
        for sid, rep in zip(ids, reps):
            sp = "val" if rep in final_val_clusters else "train"
            out.write(f"{sid}\t{rep}\t{sp}\n")

    # ---- 5. assertions ----
    assert not (final_val_clusters & condemned)
    assert n_val_final == sum(1 for sid, rep in zip(ids, reps) if rep in final_val_clusters)

    with open(args.report, "w") as r:
        r.write(f"# 09_purify_val - {args.domain}\n\n")
        r.write(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        r.write(f"- search parameters: `mmseqs search --search-type 3 --strand 1 -s {args.sens} "
                f"--min-seq-id {args.min_seq_id} -c {args.cov} --cov-mode 0 -e 1e-3 "
                f"--max-seqs 300 --split-memory-limit {args.split_memory_limit}`\n")
        r.write(f"- search target: all {n_seq:,} sequences (train \u222a val); only "
                f"cross-cluster hits count as leakage\n\n")
        r.write("| stage | val sequences | val clusters | share of all |\n|---|---:|---:|---:|\n")
        r.write(f"| after cluster split | {n_val0:,} | {len(val_clusters):,} | {n_val0/n_seq:.2%} |\n")
        r.write(f"| capped by cluster (max {args.max_val:,}) | {n_val_capped:,} | "
                f"{len(kept_clusters):,} | {n_val_capped/n_seq:.2%} |\n")
        r.write(f"| **after purification (final)** | **{n_val_final:,}** | "
                f"**{len(final_val_clusters):,}** | {n_val_final/n_seq:.2%} |\n\n")
        r.write(f"- val clusters moved back to train in full for \u2265{float(args.min_seq_id):.0%} "
                f"cross-cluster homology: {len(condemned):,} "
                f"({n_val_capped-n_val_final:,} sequences)\n")
        r.write(f"- cross-cluster homologous hit pairs: {n_hits:,}\n")
        r.write(f"- no training data is lost: every sequence removed from val goes into train\n")
        r.write(f"- output: `{args.out}`\n")
    print(open(args.report).read())
    print(f"[{args.domain}] DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
