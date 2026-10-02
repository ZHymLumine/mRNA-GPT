#!/usr/bin/env python3
"""Protein-level audit of a nucleotide-clustered property split.

The pretraining corpora and both SFT property datasets are clustered on the
NUCLEOTIDE sequence (`mmseqs cluster --dbtype 2 --min-seq-id 0.5 -c 0.8
--cluster-mode 1 -s 7.5`), because that is the level the model actually
tokenizes. Protein identity is the stricter criterion though: two paralogs can sit at 45%
nucleotide identity -- invisible to a 50% nucleotide threshold -- while their
translations align at 70%.

So this script does not re-split anything. It measures, on a split that is
already nucleotide-clean by construction, how much homology survives when you
look at the translations instead:

    mmseqs search --dbtype 1 --min-seq-id 0.3 -c 0.8 -s 7.5

Reporting it for the original CSV split and the re-split side by side is what
makes the number interpretable: some protein-level relatedness between train
and test is unavoidable in any real dataset (all of these are single-genome
paralog families), and the question is whether the re-split reduced it, not
whether it reached zero.

Both property datasets are audited the same way so the two are comparable:

    python sft/protein_homology_audit.py --name fungal_expression \
        --report reports/fungal_protein_audit.md
    python sft/protein_homology_audit.py --name mrna_stability \
        --report reports/mrna_stability_protein_audit.md
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrnagpt.vocab import GENETIC_CODE                       # noqa: E402
from sft.prepare_property_split import mmseqs_bin, run       # noqa: E402


def translate_cds(seq: str) -> str:
    """CDS -> amino acids, stopping at the first stop codon.

    Stopping (rather than encoding stops as a residue) is what a translating
    ribosome does, and it keeps the query free of characters MMseqs2's protein
    alphabet would have to guess at.
    """
    seq = seq.upper().replace("T", "U")
    aa = []
    for i in range(0, len(seq) - len(seq) % 3, 3):
        r = GENETIC_CODE.get(seq[i:i + 3])
        if r is None:
            aa.append("X")
        elif r == "*":
            break
        else:
            aa.append(r)
    return "".join(aa)


def load_rows(out_dir: str, name: str):
    rows = []
    for sp in ("train", "val", "test"):
        path = os.path.join(out_dir, f"{name}_{sp}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- run sft/prepare_property_split.py first")
        for r in csv.DictReader(open(path)):
            r["split_new"] = sp
            rows.append(r)
    return rows


def write_prot_fasta(rows, path, predicate) -> int:
    n = 0
    with open(path, "w") as fh:
        for r in rows:
            if not predicate(r):
                continue
            p = translate_cds(r["Sequence"])
            if len(p) < 10:          # too short to align meaningfully
                continue
            fh.write(f">{r['seq_id']}\n{p}\n")
            n += 1
    return n


def audit(rows, work, mm, args, split_key: str, tag: str) -> dict:
    """Fraction of held-out proteins with a >=min-seq-id homolog among train."""
    heldout = lambda r: r.get(split_key) in ("val", "test")   # noqa: E731
    train = lambda r: r.get(split_key) == "train"             # noqa: E731
    q_fa, t_fa = f"{work}/{tag}_q.fa", f"{work}/{tag}_t.fa"
    n_q = write_prot_fasta(rows, q_fa, heldout)
    n_t = write_prot_fasta(rows, t_fa, train)
    if not n_q or not n_t:
        return {"n_heldout": n_q, "n_train": n_t, "skipped": True}

    q, t = f"{work}/{tag}_qDB", f"{work}/{tag}_tDB"
    res, out = f"{work}/{tag}_res", f"{work}/{tag}_hits.tsv"
    run([mm, "createdb", q_fa, q, "--dbtype", "1", "-v", "1"])
    run([mm, "createdb", t_fa, t, "--dbtype", "1", "-v", "1"])
    run([mm, "search", q, t, res, f"{work}/tmp_{tag}",
         "-s", args.sens, "--min-seq-id", args.min_seq_id,
         "-c", args.cov, "--cov-mode", "0", "-e", args.evalue, "--max-seqs", "300",
         "--split-memory-limit", args.split_memory_limit,
         "--threads", str(args.threads), "--remove-tmp-files", "1", "-v", "1"])
    run([mm, "convertalis", q, t, res, out,
         "--format-output", "query,target,pident", "--threads", str(args.threads)])

    hit_ids, pidents = set(), []
    with open(out) as fh:
        for line in fh:
            a, _b, p = line.rstrip("\n").split("\t")
            hit_ids.add(a)
            pidents.append(float(p))
    return {"n_heldout": n_q, "n_train": n_t, "n_leaked": len(hit_ids),
            "pct": 100.0 * len(hit_ids) / n_q, "n_pairs": len(pidents),
            "max_pident": max(pidents, default=0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="e.g. mrna_stability | fungal_expression")
    ap.add_argument("--data-dir", default="sft/data")
    ap.add_argument("--report", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--min-seq-id", default="0.3",
                    help="protein identity threshold; 0.3 is the conventional "
                         "homology floor and much stricter than 50% nucleotide")
    ap.add_argument("--cov", default="0.8")
    ap.add_argument("--sens", default="7.5")
    ap.add_argument("--evalue", default="1e-3")
    ap.add_argument("--split-memory-limit", default="32G")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--mmseqs", default=None)
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    mm = mmseqs_bin(args.mmseqs)
    work = args.work or os.path.join(args.data_dir, f"_prot_{args.name}")
    os.makedirs(work, exist_ok=True)
    rows = load_rows(args.data_dir, args.name)
    print(f"{args.name}: {len(rows):,} sequences; mmseqs {mm}", flush=True)

    results = {}
    has_old = any(r.get("split_old") for r in rows)
    if has_old:
        results["original"] = audit(rows, work, mm, args, "split_old", "old")
        r = results["original"]
        print(f"  original split: {r.get('n_leaked', 0):,}/{r['n_heldout']:,} "
              f"({r.get('pct', 0):.2f}%)", flush=True)
    results["resplit"] = audit(rows, work, mm, args, "split_new", "new")
    r = results["resplit"]
    print(f"  re-split:       {r.get('n_leaked', 0):,}/{r['n_heldout']:,} "
          f"({r.get('pct', 0):.2f}%)", flush=True)

    thr = f"{float(args.min_seq_id):.0%}"
    md = [f"# {args.name}: protein-level homology audit (MMseqs2 {thr} amino-acid identity)", "",
          "The split itself is clustered at the **nucleotide** level "
          "(`--min-seq-id 0.5 -c 0.8 --cluster-mode 1`, matching pretraining and the "
          "level the model actually tokenizes). This file does not re-split anything; "
          "it translates the CDS into protein and **checks once more**: paralogs that "
          "50% nucleotide identity cannot catch (45% nucleotide / 70% protein identity "
          "is a common combination) show up here.", "",
          f"Test: `mmseqs search --dbtype 1 --min-seq-id {args.min_seq_id} "
          f"-c {args.cov} -s {args.sens} -e {args.evalue}`, query = the translations of "
          f"val u test, target = the translations of train.", "",
          "| split | val/test proteins | train proteins | with a >="
          f"{thr} protein homolog in train | fraction | max identity |",
          "|---|---:|---:|---:|---:|---:|"]
    for key, label in (("original", "the split shipped with the CSV"),
                       ("resplit", "**this work's homology-aware split**")):
        if key not in results:
            continue
        r = results[key]
        if r.get("skipped"):
            md.append(f"| {label} | {r['n_heldout']:,} | {r['n_train']:,} | – | – | – |")
            continue
        md.append(f"| {label} | {r['n_heldout']:,} | {r['n_train']:,} | "
                  f"{r['n_leaked']:,} | {r['pct']:.2f}% | {r['max_pident']:.1f}% |")
    md += ["", "## How to read this number", "",
           f"Protein {thr} is the conventional lower bound for homology and is far "
           "stricter than 50% nucleotide identity, so **a non-zero residue is "
           "expected** -- paralog families within a single genome cannot be separated "
           "completely without destroying the size of the dataset. The meaningful "
           "comparison is the change in this column between **the original split and "
           "the re-split** under the same ruler, not how far it is from zero."]
    if "original" in results and not results["resplit"].get("skipped"):
        o, n = results["original"], results["resplit"]
        if not o.get("skipped"):
            delta = o["pct"] - n["pct"]
            md += ["", f"This dataset: {o['pct']:.2f}% -> {n['pct']:.2f}% "
                       f"({'down' if delta > 0 else 'up'} {abs(delta):.2f} percentage points)."]
    md += ["", f"For the nucleotide-level counterpart see `reports/{args.name}_leakage.md` "
               "(that column is 0.00%, which holds by construction).", ""]
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as fh:
        fh.write("\n".join(md))
    print(open(args.report).read())
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    print(f"{args.name}: DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
