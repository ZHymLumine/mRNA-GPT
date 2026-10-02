#!/usr/bin/env python3
"""Homology-aware re-split of an experimental-property CSV (Sequence, Value, Split).

Same methodology as the pretraining corpora (scripts/02_linclust.sh ->
03_split_by_cluster.py -> 09_purify_val.py), applied to a small measured-property
dataset: exact dedup, MMseqs2 connected-component clustering at 50% identity /
80% bidirectional coverage, whole-cluster assignment to train/val/test, then an
exhaustive sensitive search that moves any cluster with residual cross-cluster
homology back into train.

This generalizes what was done once for Fungal_expression.csv
(reports/fungal_expression_leakage.md) so the mRNA-stability dataset -- and any
later property dataset -- goes through the identical, scripted procedure.

Two leakage sources are measured on the dataset's own Split column before
re-splitting:

  exact     the SAME sequence appearing in more than one of train/val/test.
            Fungal_expression.csv had none; mRNA_Stability.csv is built from
            replicate measurements, so it has many, and they are pure
            train-on-test.
  homology  a val/test sequence with a >=50% identity homolog in train.

Replicate rows are collapsed to one row per distinct sequence, with Value the
mean of its measurements (n_measurements / value_sd are kept so the spread is
auditable). Collapsing has to happen before clustering: leaving replicates in
would let the same sequence land on both sides of the split, which is exactly
the leakage this file exists to remove.

Filtering mirrors scripts/01_extract_cds.py: alphabet and frame are hard
requirements, while ATG-start / stop-end / internal-stop are RECORDED rather
than filtered, so the SFT corpus is treated the same way the pretraining corpus
was.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict

import numpy as np

# MMseqs2 is third-party software and is not included in this repository:
# install it separately. The binary is taken from $PATH unless MMSEQS_BIN or
# --mmseqs overrides it.
DEFAULT_MMSEQS = "mmseqs"
STOPS = ("UAA", "UAG", "UGA")
ALPHABET = set("ACGU")


def mmseqs_bin(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.environ.get("MMSEQS_BIN"):
        return os.environ["MMSEQS_BIN"]
    return shutil.which("mmseqs") or DEFAULT_MMSEQS


def run(cmd, quiet=True):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True,
                   stdout=subprocess.DEVNULL if quiet else None)


# --------------------------------------------------------------------------- #
# 1. load, filter, dedup
# --------------------------------------------------------------------------- #
def load_and_dedup(csv_path: str, name: str, max_codons: int,
                   seq_col: str = "Sequence", value_col: str = "Value"):
    """-> (records, stats). One record per distinct sequence, input order kept."""
    order: list[str] = []
    vals: dict[str, list[float]] = defaultdict(list)
    orig_splits: dict[str, list[str]] = defaultdict(list)
    stats = Counter()

    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            stats["rows"] += 1
            if not row.get(seq_col) or not row.get(value_col):
                stats["missing_field"] += 1
                continue
            seq = row[seq_col].strip().upper().replace("T", "U")
            if set(seq) - ALPHABET:
                stats["bad_char"] += 1
                continue
            if len(seq) % 3:
                stats["bad_frame"] += 1
                continue
            n_codon = len(seq) // 3
            if n_codon == 0 or n_codon > max_codons:
                stats["bad_length"] += 1
                continue
            stats["kept_rows"] += 1
            if seq not in vals:
                order.append(seq)
            vals[seq].append(float(row[value_col]))
            if row.get("Split"):
                orig_splits[seq].append(row["Split"])

    records = []
    for i, seq in enumerate(order):
        v = vals[seq]
        sp = orig_splits.get(seq, [])
        records.append({
            "seq_id": f"{name}_{i:06d}",
            "Sequence": seq,
            "Value": statistics.mean(v),
            "n_measurements": len(v),
            "value_sd": statistics.stdev(v) if len(v) > 1 else 0.0,
            "split_old": Counter(sp).most_common(1)[0][0] if sp else "",
            "split_old_all": "|".join(sorted(set(sp))),
        })
    stats["unique"] = len(records)
    stats["replicated"] = sum(1 for r in records if r["n_measurements"] > 1)
    stats["multi_split_seqs"] = sum(1 for r in records if "|" in r["split_old_all"])
    stats["multi_split_rows"] = sum(r["n_measurements"] for r in records
                                    if "|" in r["split_old_all"])
    codons = [len(r["Sequence"]) // 3 for r in records]
    stats["starts_aug"] = sum(r["Sequence"][:3] == "AUG" for r in records)
    stats["ends_stop"] = sum(r["Sequence"][-3:] in STOPS for r in records)
    stats["internal_stop"] = sum(
        any(r["Sequence"][j:j + 3] in STOPS for j in range(0, len(r["Sequence"]) - 3, 3))
        for r in records)
    stats["codon_mean"] = int(statistics.mean(codons))
    stats["codon_median"] = int(statistics.median(codons))
    stats["codon_max"] = max(codons)
    return records, stats


def write_fasta(records, path, ids=None):
    keep = None if ids is None else set(ids)
    n = 0
    with open(path, "w") as fh:
        for r in records:
            if keep is not None and r["seq_id"] not in keep:
                continue
            fh.write(f">{r['seq_id']}\n{r['Sequence']}\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# 2. clustering
# --------------------------------------------------------------------------- #
def cluster(fasta: str, work: str, mm: str, args) -> dict[str, str]:
    """-> {member_id: representative_id} (connected components)."""
    db, clu, tsv = f"{work}/DB", f"{work}/DB_clu", f"{work}/clusters.tsv"
    run([mm, "createdb", fasta, db, "--dbtype", "2", "-v", "1"])
    run([mm, "cluster", db, clu, f"{work}/tmp_clu",
         "--min-seq-id", args.min_seq_id, "-c", args.cov, "--cov-mode", "0",
         "--cluster-mode", "1", "-s", args.sens,
         "--split-memory-limit", args.split_memory_limit,
         "--threads", str(args.threads), "--remove-tmp-files", "1", "-v", "1"])
    run([mm, "createtsv", db, db, clu, tsv, "--threads", str(args.threads)])
    member2rep = {}
    with open(tsv) as fh:
        for line in fh:
            rep, mem = line.rstrip("\n").split("\t")
            member2rep[mem] = rep
    return member2rep


# --------------------------------------------------------------------------- #
# 3. homology search helper
# --------------------------------------------------------------------------- #
def search_hits(query_fa: str, target_fa: str, work: str, tag: str, mm: str, args):
    """-> list of (query_id, target_id, pident) at >=min_seq_id / >=cov."""
    q, t = f"{work}/{tag}_qDB", f"{work}/{tag}_tDB"
    res, out = f"{work}/{tag}_res", f"{work}/{tag}_hits.tsv"
    run([mm, "createdb", query_fa, q, "--dbtype", "2", "-v", "1"])
    run([mm, "createdb", target_fa, t, "--dbtype", "2", "-v", "1"])
    run([mm, "search", q, t, res, f"{work}/tmp_{tag}",
         "--search-type", "3", "--strand", "1", "-s", args.sens,
         "--min-seq-id", args.min_seq_id, "-c", args.cov, "--cov-mode", "0",
         "-e", "1e-3", "--max-seqs", "300",
         "--split-memory-limit", args.split_memory_limit,
         "--threads", str(args.threads), "--remove-tmp-files", "1", "-v", "1"])
    run([mm, "convertalis", q, t, res, out,
         "--format-output", "query,target,pident", "--threads", str(args.threads)])
    hits = []
    with open(out) as fh:
        for line in fh:
            a, b, p = line.rstrip("\n").split("\t")
            hits.append((a, b, float(p)))
    return hits


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seq-col", default="Sequence",
                    help="CDS column in --csv (e.g. CDS_sequence for the E. coli TE file)")
    ap.add_argument("--value-col", default="Value",
                    help="measured-property column in --csv (e.g. TE)")
    ap.add_argument("--name", required=True,
                    help="output basename, e.g. mrna_stability -> "
                         "mrna_stability_{train,val,test}.csv")
    ap.add_argument("--out-dir", default="sft/data")
    ap.add_argument("--report", required=True)
    ap.add_argument("--work", default=None, help="mmseqs scratch (default: <out-dir>/_work_<name>)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--overshoot", type=float, default=0.05,
                    help="fraction a giant cluster may push val/test past its target "
                         "before it is skipped instead")
    ap.add_argument("--max-codons", type=int, default=2044,
                    help="block_size 2048 minus [BOS]/[EOS]")
    ap.add_argument("--min-seq-id", default="0.5")
    ap.add_argument("--cov", default="0.8")
    ap.add_argument("--sens", default="7.5")
    ap.add_argument("--split-memory-limit", default="32G")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mmseqs", default=None)
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--skip-old-split-leakage", action="store_true",
                    help="skip the (slow) measurement of the input CSV's own leakage")
    args = ap.parse_args()

    t0 = time.time()
    mm = mmseqs_bin(args.mmseqs)
    os.makedirs(args.out_dir, exist_ok=True)
    work = args.work or os.path.join(args.out_dir, f"_work_{args.name}")
    os.makedirs(work, exist_ok=True)
    print(f"mmseqs: {mm}\nwork:   {work}", flush=True)

    # ---- 1. load / filter / dedup ----
    records, stats = load_and_dedup(args.csv, args.name, args.max_codons,
                                    args.seq_col, args.value_col)
    by_id = {r["seq_id"]: r for r in records}
    print(f"[{args.name}] {stats['rows']:,} rows -> {stats['kept_rows']:,} passed filters "
          f"-> {stats['unique']:,} distinct sequences "
          f"({stats['replicated']:,} had replicate measurements)", flush=True)
    all_fa = f"{work}/all.fa"
    write_fasta(records, all_fa)

    # ---- 2. leakage of the dataset's own split, for the record ----
    old = {}
    if not args.skip_old_split_leakage and any(r["split_old"] for r in records):
        old_tr = [r["seq_id"] for r in records if r["split_old"] == "train"]
        old_ht = [r["seq_id"] for r in records if r["split_old"] in ("val", "test")]
        if old_tr and old_ht:
            write_fasta(records, f"{work}/old_train.fa", old_tr)
            write_fasta(records, f"{work}/old_heldout.fa", old_ht)
            hits = search_hits(f"{work}/old_heldout.fa", f"{work}/old_train.fa",
                               work, "old", mm, args)
            leaked = {q for q, _t, _p in hits}
            old = {"n_train": len(old_tr), "n_heldout": len(old_ht),
                   "n_leaked": len(leaked),
                   "pct": 100.0 * len(leaked) / len(old_ht),
                   "max_pident": max((p for _q, _t, p in hits), default=0.0)}
            print(f"[{args.name}] original split: {old['n_leaked']:,}/{old['n_heldout']:,} "
                  f"({old['pct']:.2f}%) held-out sequences have a >={float(args.min_seq_id):.0%} "
                  f"homolog in train (max identity {old['max_pident']:.1f}%)", flush=True)

    # ---- 3. cluster ----
    member2rep = cluster(all_fa, work, mm, args)
    missing = [r["seq_id"] for r in records if r["seq_id"] not in member2rep]
    if missing:
        raise SystemExit(f"{len(missing)} sequences absent from the cluster tsv, e.g. {missing[:3]}")
    members_of = defaultdict(list)
    for sid, rep in member2rep.items():
        members_of[rep].append(sid)
    reps = sorted(members_of)
    biggest = max(len(v) for v in members_of.values())
    print(f"[{args.name}] {len(records):,} sequences -> {len(reps):,} clusters "
          f"(largest {biggest:,} = {100*biggest/len(records):.2f}%)", flush=True)

    # ---- 4. whole-cluster train/val/test assignment ----
    rng = np.random.default_rng(args.seed)
    n = len(records)
    targets = {"val": int(n * args.val_frac), "test": int(n * args.test_frac)}
    caps = {k: int(v * (1 + args.overshoot)) for k, v in targets.items()}
    split_of_rep: dict[str, str] = {r: "train" for r in reps}
    filled = {"val": 0, "test": 0}
    for i in rng.permutation(len(reps)):
        rep = reps[i]
        size = len(members_of[rep])
        for which in ("val", "test"):
            if filled[which] >= targets[which]:
                continue
            if filled[which] + size > caps[which]:
                continue          # a giant cluster would overshoot: skip, don't split it
            split_of_rep[rep] = which
            filled[which] += size
            break
        if all(filled[w] >= targets[w] for w in ("val", "test")):
            break
    post_cluster = Counter(split_of_rep[member2rep[r["seq_id"]]] for r in records)
    print(f"[{args.name}] after cluster split: {dict(post_cluster)}", flush=True)

    # ---- 5. purify: any residual cross-cluster homology condemns the whole cluster ----
    heldout_ids = [r["seq_id"] for r in records
                   if split_of_rep[member2rep[r["seq_id"]]] in ("val", "test")]
    write_fasta(records, f"{work}/heldout.fa", heldout_ids)
    hits = search_hits(f"{work}/heldout.fa", all_fa, work, "purify", mm, args)
    condemned = set()
    n_cross = 0
    for q, t, _p in hits:
        rq = member2rep[q]
        if member2rep[t] == rq:
            continue              # same-cluster hits are expected, not leakage
        n_cross += 1
        condemned.add(rq)
        if split_of_rep[member2rep[t]] in ("val", "test"):
            condemned.add(member2rep[t])
    for rep in condemned:
        split_of_rep[rep] = "train"
    n_moved = sum(len(members_of[r]) for r in condemned)
    print(f"[{args.name}] purify: {n_cross:,} cross-cluster hits -> {len(condemned):,} "
          f"clusters ({n_moved:,} sequences) moved back to train", flush=True)

    # ---- 6. verify residual leakage is zero ----
    final_heldout = [r["seq_id"] for r in records
                     if split_of_rep[member2rep[r["seq_id"]]] in ("val", "test")]
    write_fasta(records, f"{work}/heldout2.fa", final_heldout)
    hits2 = search_hits(f"{work}/heldout2.fa", all_fa, work, "verify", mm, args)
    residual = {q for q, t, _p in hits2 if member2rep[t] != member2rep[q]}
    print(f"[{args.name}] verification: {len(residual)} of {len(final_heldout):,} held-out "
          f"sequences still have a cross-cluster homolog", flush=True)
    if residual:
        raise SystemExit(f"purification did not converge: {sorted(residual)[:5]}")

    # ---- 7. write ----
    final = Counter()
    changed = 0
    for r in records:
        r["split_new"] = split_of_rep[member2rep[r["seq_id"]]]
        r["cluster_rep"] = member2rep[r["seq_id"]]
        final[r["split_new"]] += 1
        if r["split_old"] and r["split_old"] != r["split_new"]:
            changed += 1
    cols = ["seq_id", "Sequence", "Value", "n_measurements", "value_sd",
            "split_old", "split_old_all", "split_new"]
    for sp in ("train", "val", "test"):
        path = os.path.join(args.out_dir, f"{args.name}_{sp}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in records:
                if r["split_new"] == sp:
                    w.writerow(r)
        print(f"wrote {final[sp]:,} -> {path}")

    clu_path = os.path.join(args.out_dir, f"{args.name}_clusters.tsv.gz")
    with gzip.open(clu_path, "wt") as fh:
        for r in records:
            fh.write(f"{r['cluster_rep']}\t{r['seq_id']}\n")
    split_path = os.path.join(args.out_dir, f"{args.name}_split_final.tsv.gz")
    with gzip.open(split_path, "wt") as fh:
        fh.write("seq_id\tcluster_rep\tsplit\n")
        for r in records:
            fh.write(f"{r['seq_id']}\t{r['cluster_rep']}\t{r['split_new']}\n")

    # ---- 8. report ----
    vals = [r["Value"] for r in records]
    md = [f"# {os.path.basename(args.csv)}: homology-aware re-split (MMseqs2 "
          f"{float(args.min_seq_id):.0%})", "",
          f"Source: `{args.csv}`", "",
          "## Input filtering and exact deduplication", "",
          "| | count |", "|---|---:|",
          f"| raw rows | {stats['rows']:,} |",
          f"| passing the filters (ACGU / reading frame / <={args.max_codons} codons) | {stats['kept_rows']:,} |",
          f"| **distinct sequences after exact deduplication** | **{stats['unique']:,}** |",
          f"| of those, sequences with replicate measurements | {stats['replicated']:,} |", "",
          f"Repeated measurements of the same sequence are averaged (`n_measurements` / `value_sd` are kept in the output CSV).",
          f"Codon length mean {stats['codon_mean']} / median {stats['codon_median']} / "
          f"max {stats['codon_max']}; AUG start "
          f"{100*stats['starts_aug']/stats['unique']:.2f}%, stop-codon end "
          f"{100*stats['ends_stop']/stats['unique']:.2f}%, internal stop codon present "
          f"{100*stats['internal_stop']/stats['unique']:.2f}%"
          f" (consistent with `scripts/01_extract_cds.py`: these three are recorded only, not filtered on).",
          f"Value range {min(vals):.4f} ~ {max(vals):.4f}, mean {statistics.mean(vals):.4f}.", ""]

    if stats["multi_split_seqs"]:
        md += ["## Leakage in the original split, (1) exact duplicates", "",
               f"**{stats['multi_split_seqs']:,} sequences appear in more than one split of the original `Split` column**"
               f" (covering {stats['multi_split_rows']:,} rows) -- the same sequence is in train and in "
               f"val/test, which is train-on-test in its most direct form. A dataset with replicate "
               f"measurements that is not deduplicated by sequence before splitting is bound to end up "
               f"this way.", ""]
    if old:
        md += ["## Leakage in the original split, (2) homology", "",
               f"Using the same MMseqs2 test as pretraining (`--min-seq-id {args.min_seq_id} "
               f"-c {args.cov} -s {args.sens}`, after deduplication and under the original split): "
               f"**{old['pct']:.2f}% ({old['n_leaked']:,}/{old['n_heldout']:,}) of the val/test "
               f"sequences have a homolog at >={float(args.min_seq_id):.0%} identity in train**, "
               f"up to {old['max_pident']:.1f}%.", ""]

    md += ["## Re-split", "",
           f"`mmseqs cluster --min-seq-id {args.min_seq_id} -c {args.cov} "
           f"--cluster-mode 1 -s {args.sens}` (connected-component clustering, {len(records):,} sequences -> "
           f"{len(reps):,} clusters, the largest holding {100*biggest/len(records):.2f}%), "
           f"whole clusters assigned at random to train/val/test (seed {args.seed}, target "
           f"{100*(1-args.val_frac-args.test_frac):.0f}%/{100*args.val_frac:.0f}%/"
           f"{100*args.test_frac:.0f}%), then purified with an exhaustive `mmseqs search`:",
           "", f"- {n_cross:,} cross-cluster homology hits, involving {len(condemned):,} val/test clusters"
                f" ({n_moved:,} sequences) -> the whole cluster is moved back to train",
           f"- a repeat search after purification confirms **0 residual leakage** ({len(final_heldout):,} val/test "
           f"sequences have no >={float(args.min_seq_id):.0%} hit against any other cluster)", "",
           "## Final result", "",
           "| | train | val | test |", "|---|---:|---:|---:|"]
    if any(r["split_old"] for r in records):
        oc = Counter(r["split_old"] for r in records if r["split_old"])
        md.append(f"| original CSV (deduplicated, under the original split) | {oc.get('train', 0):,} | "
                  f"{oc.get('val', 0):,} | {oc.get('test', 0):,} |")
    md += [f"| after whole-cluster splitting | {post_cluster['train']:,} | {post_cluster['val']:,} | "
           f"{post_cluster['test']:,} |",
           f"| **after purification (the split actually used)** | **{final['train']:,}** | **{final['val']:,}** | "
           f"**{final['test']:,}** |", ""]
    if old:
        md += [f"**Leakage rate at >={float(args.min_seq_id):.0%} identity: {old['pct']:.2f}% -> 0.00%**", ""]
    if any(r["split_old"] for r in records):
        md += [f"Compared with the original split, {100*changed/stats['unique']:.1f}% "
               f"({changed:,}/{stats['unique']:,}) of the sequences were reassigned.", ""]
    md += ["## Outputs", "",
           f"- `{args.out_dir}/{args.name}_{{train,val,test}}.csv` -- the final split, columns "
           f"`{', '.join(cols)}`",
           f"- `{clu_path}` -- MMseqs2 clustering result (rep, member)",
           f"- `{split_path}` -- the final seq_id -> split mapping", "",
           "Everything downstream of this dataset (selecting the SFT training set, training the "
           "LightGBM predictor, the real-gene-anchored codon-profile comparison) uses this split "
           "rather than the `Split` column shipped with the CSV.", ""]
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as fh:
        fh.write("\n".join(md))
    print(open(args.report).read())

    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    print(f"[{args.name}] DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
