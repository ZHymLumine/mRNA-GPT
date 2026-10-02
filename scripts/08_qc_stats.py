#!/usr/bin/env python3
"""
08 - Summarise dataset composition and CDS syntactic validity from meta.tsv.gz

This supports:
  * quantifying the proportion of syntactically valid full-length CDS -- what is
    reported here is the baseline of the training data itself, against which the
    same statistics for generated sequences should be compared;
  * the description of pretraining data size / species coverage in the paper.
"""
import argparse
import gzip
from collections import Counter

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--split", help="split.tsv.gz; when given, statistics are reported per train/val")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    split_of = {}
    if args.split:
        with gzip.open(args.split, "rt") as fh:
            fh.readline()
            for line in fh:
                sid, _rep, sp = line.rstrip("\n").split("\t")
                split_of[sid] = sp

    groups = ["all"] + (["train", "val"] if split_of else [])
    stat = {g: Counter() for g in groups}
    lens = {g: [] for g in groups}
    species = {g: set() for g in groups}
    accs = {g: set() for g in groups}
    phyla = {g: Counter() for g in groups}

    with gzip.open(args.meta, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(hdr)}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            gs = ["all"]
            if split_of:
                gs.append(split_of[p[idx["seq_id"]]])
            atg = p[idx["starts_atg"]] == "1"
            stop = p[idx["ends_stop"]] == "1"
            internal = p[idx["internal_stop"]] == "1"
            n_codon = int(p[idx["n_codon"]])
            for g in gs:
                c = stat[g]
                c["n"] += 1
                c["atg"] += atg
                c["stop"] += stop
                c["internal"] += internal
                c["valid"] += (atg and stop and not internal)
                c["dup"] += int(p[idx["dup_count"]]) > 1
                lens[g].append(n_codon)
                species[g].add(p[idx["species"]])
                accs[g].add(p[idx["accession"]])
                phyla[g][p[idx["phylum"]]] += 1

    with open(args.report, "w") as r:
        r.write(f"# 08_qc_stats - {args.domain}\n\n")
        r.write("## Dataset composition\n\n")
        r.write("| | sequences | assemblies | species | phyla | codon length P50/P90/max |\n")
        r.write("|---|---:|---:|---:|---:|---|\n")
        for g in groups:
            L = np.asarray(lens[g])
            r.write(f"| {g} | {stat[g]['n']:,} | {len(accs[g]):,} | "
                    f"{len(species[g] - {''}):,} | {len(set(phyla[g]) - {''})} | "
                    f"{int(np.percentile(L,50))} / {int(np.percentile(L,90))} / {int(L.max())} |\n")

        r.write("\n## CDS syntactic validity (training-data baseline, for comparison with generated sequences)\n\n")
        r.write("| | starts with ATG | ends with a stop codon | has an internal stop codon | **all three satisfied** |\n")
        r.write("|---|---:|---:|---:|---:|\n")
        for g in groups:
            c = stat[g]
            n = max(c["n"], 1)
            r.write(f"| {g} | {c['atg']/n:.2%} | {c['stop']/n:.2%} | "
                    f"{c['internal']/n:.2%} | **{c['valid']/n:.2%}** |\n")
        r.write("\nNote: this dataset is **not filtered on these flags** (matching the "
                "original processing in the paper); they are only recorded. NCBI's "
                "`_cds_from_genomic.fna` inherently contains some partial CDS, genes "
                "using alternative start codons, and annotations such as "
                "selenocysteine that produce an \"internal stop codon\".\n")

        r.write(f"\n## Exact duplicates\n\n")
        for g in groups:
            r.write(f"- {g}: {stat[g]['dup']:,} sequences had an identical copy "
                    f"({stat[g]['dup']/max(stat[g]['n'],1):.2%})\n")

        r.write("\n## Phylum distribution (top 10)\n\n")
        r.write("| phylum | sequences | share |\n|---|---:|---:|\n")
        tot = stat["all"]["n"]
        for ph, n in phyla["all"].most_common(10):
            r.write(f"| {ph or '(unannotated)'} | {n:,} | {n/tot:.2%} |\n")

    print(open(args.report).read())


if __name__ == "__main__":
    main()
