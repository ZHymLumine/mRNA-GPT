#!/usr/bin/env python3
"""
06 - Quantify the residual homology leakage between train and val, and compare it
with the old random 90/10 split.

Three sets of numbers are produced per domain:
  A. New split (cluster-aware): sample from val -> mmseqs search against train ->
     the fraction with a hit at >=50% / >=30% identity.
  B. Old-split control (random 90/10, seed=42): the same deduplicated sequences
     split purely at random, run through the same search.
  C. Exact-duplicate statistics.

A has to be measured rather than assumed to be zero by construction: linclust is
a greedy approximation and can miss some homologous pairs.  This step measures
exactly how many it misses.
"""
import argparse
import gzip
import os
import shutil
import subprocess
import sys
import time

import numpy as np


def run(cmd, **kw):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def read_split(path):
    train, val = [], []
    with gzip.open(path, "rt") as fh:
        fh.readline()
        for line in fh:
            sid, _rep, sp = line.rstrip("\n").split("\t")
            (val if sp == "val" else train).append(sid)
    return train, val


def write_subset(cds_gz, wanted, out_fa):
    """Extract the sequences with the given ids from cds.fasta.gz. wanted is a set[str]."""
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


def max_pident(query_fa, target_fa, work, threads, sens, min_seq_id, cov,
               split_memory_limit="0"):
    """For each query, return its highest sequence identity against the target
    (0-1; no hit counts as 0)."""
    os.makedirs(work, exist_ok=True)
    qdb, tdb, res = f"{work}/qDB", f"{work}/tDB", f"{work}/res"
    tsv = f"{work}/hits.tsv"
    if not os.path.exists(tsv):
        run(["mmseqs", "createdb", query_fa, qdb, "--dbtype", "2"],
            stdout=subprocess.DEVNULL)
        run(["mmseqs", "createdb", target_fa, tdb, "--dbtype", "2"],
            stdout=subprocess.DEVNULL)
        run(["mmseqs", "search", qdb, tdb, res, f"{work}/tmp",
             "--search-type", "3", "--strand", "1",
             "-s", sens, "--min-seq-id", min_seq_id, "-c", cov, "--cov-mode", "0",
             "-e", "1e-3", "--max-seqs", "300",
             "--split-memory-limit", split_memory_limit,
             "--threads", threads, "--remove-tmp-files", "1"],
            stdout=subprocess.DEVNULL)
        run(["mmseqs", "convertalis", qdb, tdb, res, tsv,
             "--format-output", "query,target,pident", "--threads", threads],
            stdout=subprocess.DEVNULL)
    else:
        print(f"  (reusing existing {tsv})", flush=True)

    best = {}
    raw_max = 0.0
    with open(tsv) as fh:
        for line in fh:
            q, t, pid = line.rstrip("\n").split("\t")
            if q == t:
                continue  # the same sequence (should not happen: train and val are disjoint)
            pid = float(pid)
            raw_max = max(raw_max, pid)
            if pid > best.get(q, 0.0):
                best[q] = pid
    # Across MMseqs2 versions pident is reported either as 0-1 or as 0-100;
    # decide from the observed maximum and normalise everything to 0-1.
    if raw_max > 1.0:
        best = {q: v / 100.0 for q, v in best.items()}
    print(f"  raw pident maximum {raw_max:.4f} -> "
          f"{'normalised from percent' if raw_max > 1.0 else 'already a 0-1 fraction'}", flush=True)
    return best


def summarize(best, n_query):
    arr = np.zeros(n_query)
    vals = list(best.values())
    arr[:len(vals)] = vals
    return {
        "n_query": n_query,
        "hit_any": len(best) / n_query,
        "ge30": sum(1 for v in best.values() if v >= 0.30) / n_query,
        "ge40": sum(1 for v in best.values() if v >= 0.40) / n_query,
        "ge50": sum(1 for v in best.values() if v >= 0.50) / n_query,
        "ge70": sum(1 for v in best.values() if v >= 0.70) / n_query,
        "ge90": sum(1 for v in best.values() if v >= 0.90) / n_query,
        "ge99": sum(1 for v in best.values() if v >= 0.99) / n_query,
        "median_max_pident": float(np.median(arr)),
        "arr": arr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cds", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--work", required=True, help="working directory (node-local disk)")
    ap.add_argument("--out-dir", required=True, help="output directory for the report and figure")
    ap.add_argument("--n-query", type=int, default=100_000)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--sens", default="7.5")
    ap.add_argument("--min-seq-id", default="0.3")
    ap.add_argument("--cov", default="0.8")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-memory-limit", default="0",
                    help="per-split memory cap for the mmseqs prefilter (e.g. 200G). "
                         "0 = use all available memory, which gets OOM-killed inside a PBS cgroup")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids = read_split(args.split)
    all_ids = train_ids + val_ids
    n_all = len(all_ids)
    print(f"[{args.domain}] train={len(train_ids)} val={len(val_ids)} total={n_all}",
          flush=True)

    nq = min(args.n_query, len(val_ids))
    summaries = {}

    # ---------- A. new split (cluster-aware) ----------
    q_new = set(np.asarray(val_ids)[rng.choice(len(val_ids), nq, replace=False)].tolist())
    write_subset(args.cds, q_new, f"{args.work}/q_new.fa")
    write_subset(args.cds, set(train_ids), f"{args.work}/t_new.fa")
    print(f"[{args.domain}] A) cluster-aware split search ({time.time()-t0:.0f}s)", flush=True)
    best = max_pident(f"{args.work}/q_new.fa", f"{args.work}/t_new.fa",
                      f"{args.work}/w_new", args.threads, args.sens,
                      args.min_seq_id, args.cov, args.split_memory_limit)
    summaries["new"] = summarize(best, nq)

    # ---------- B. old-split control (random 90/10) ----------
    perm = rng.permutation(n_all)
    n_val_rand = len(val_ids)
    arr_ids = np.asarray(all_ids)
    rand_val = arr_ids[perm[:n_val_rand]]
    rand_train = set(arr_ids[perm[n_val_rand:]].tolist())
    q_old = set(rand_val[rng.choice(n_val_rand, nq, replace=False)].tolist())
    write_subset(args.cds, q_old, f"{args.work}/q_old.fa")
    write_subset(args.cds, rand_train, f"{args.work}/t_old.fa")
    print(f"[{args.domain}] B) random split search ({time.time()-t0:.0f}s)", flush=True)
    best_old = max_pident(f"{args.work}/q_old.fa", f"{args.work}/t_old.fa",
                          f"{args.work}/w_old", args.threads, args.sens,
                          args.min_seq_id, args.cov, args.split_memory_limit)
    summaries["old"] = summarize(best_old, nq)

    # ---------- C. exact-duplicate statistics ----------
    n_dup_removed = n_multi = 0
    with gzip.open(args.meta, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        i_dup = hdr.index("dup_count")
        for line in fh:
            d = int(line.rstrip("\n").split("\t")[i_dup])
            if d > 1:
                n_multi += 1
                n_dup_removed += d - 1

    # ---------- report ----------
    rep = os.path.join(args.out_dir, f"{args.domain}_leakage.md")
    with open(rep, "w") as r:
        r.write(f"# Leakage quantification - {args.domain}\n\n")
        r.write(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        r.write(f"- search parameters: `mmseqs search --search-type 3 --strand 1 -s {args.sens} "
                f"--min-seq-id {args.min_seq_id} -c {args.cov} --cov-mode 0 -e 1e-3`\n")
        r.write(f"- queries per group: {nq:,} (randomly sampled from each validation set, seed={args.seed})\n\n")
        r.write("## A/B comparison: fraction of validation sequences with a homologue in training\n\n")
        r.write("| split | any hit | ≥30% | ≥40% | **≥50%** | ≥70% | ≥90% | ≥99% |\n")
        r.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for key, label in [("old", "random 90/10 (original pipeline)"),
                           ("new", "MMseqs2 50% cluster split (this work)")]:
            s = summaries[key]
            r.write(f"| {label} | {s['hit_any']:.2%} | {s['ge30']:.2%} | {s['ge40']:.2%} | "
                    f"**{s['ge50']:.2%}** | {s['ge70']:.2%} | {s['ge90']:.2%} | "
                    f"{s['ge99']:.2%} |\n")
        red = summaries["old"]["ge50"]
        newv = summaries["new"]["ge50"]
        r.write(f"\n**Leakage at ≥50% identity drops from {red:.2%} to {newv:.2%} "
                f"(a {(1-newv/red)*100:.1f}% reduction)**\n" if red > 0 else "\n")
        r.write("\n## C. Exact duplicates\n\n")
        r.write(f"- sequences kept after deduplication: {n_all:,}\n")
        r.write(f"- sequences that had an identical copy: {n_multi:,}\n")
        r.write(f"- exact duplicate records removed: {n_dup_removed:,}\n")
        r.write(f"- exact duplicates between the new train and val: **0** "
                f"(guaranteed by construction: deduplication happens before the "
                f"split, and no cluster straddles it)\n")
    print(open(rep).read())

    # ---------- figure ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(0, 1, 51)
        for key, label, c in [("old", "Random 90/10 (original)", "#c44e52"),
                              ("new", "MMseqs2 50% cluster split (this work)", "#4c72b0")]:
            ax.hist(summaries[key]["arr"], bins=bins, alpha=0.6, label=label,
                    color=c, density=True)
        ax.axvline(0.5, ls="--", c="k", lw=1)
        ax.set_xlabel("Max sequence identity of a validation CDS to any training CDS")
        ax.set_ylabel("Density")
        ax.set_title(f"{args.domain}: train/validation homology leakage")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"{args.domain}_leakage.png"), dpi=200)
        print(f"figure -> {args.out_dir}/{args.domain}_leakage.png")
    except Exception as e:
        print(f"(plot skipped: {e})")

    np.savez(os.path.join(args.out_dir, f"{args.domain}_leakage.npz"),
             new=summaries["new"]["arr"], old=summaries["old"]["arr"])
    print(f"[{args.domain}] DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
