#!/usr/bin/env python3
"""Descriptor means for real top- and bottom-quartile genes of each property,
computed over the WHOLE quartile of the held-out TEST split.

Why the whole quartile and not a sample: these means are the reference every
"is this design realistic?" statement is read against, and for some properties
the quantity of interest -- the difference between the two quartiles -- is
smaller than the noise between two 200-gene draws. Bacterial expression is the
clear case: its real high-vs-low CAI difference is about -0.001, while
re-drawing 200 genes moves that difference by a comparable amount. Sampling
would make the sign of the headline number a coin flip, so we do not sample.

A bootstrap over genes gives the interval on each difference, which is what
makes "these descriptors do not separate high from low expression" a claim
rather than an impression.

    python sft/real_gene_reference.py --out figures/real_gene_metrics.json \
        --fasta-dir "$MRNA_GPT_RUNS/real_gene_panels"
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import tempfile

import numpy as np

from sft.reference_quartiles import load_split  # noqa: E402

TASKS = {"expression": ("sft/data/fungal_expression_test.csv", "sft/lightgbm_expression"),
         "stability": ("sft/data/mrna_stability_test.csv", "sft/lightgbm_stability"),
         "bacexp": ("sft/data/bacteria_expression_test.csv", "sft/lightgbm_bacexp")}
METRICS = ["cai", "tai", "gc", "gc3", "mfe_per_nt"]


def _codon_metrics():
    # evaluate/ is a namespace dir shadowed by HuggingFace's installed "evaluate"
    spec = importlib.util.spec_from_file_location("cm", "evaluate/codon_metrics.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def codons(seq: str) -> list[str]:
    # the shipped CAI references are keyed in RNA (U) form; scoring DNA (T)
    # codons against them silently skips every T-containing codon
    s = seq.upper().replace("T", "U")
    return [s[i:i + 3] for i in range(0, len(s) - len(s) % 3, 3)]


# ViennaRNA is third-party software and is not included in this repository:
# install it separately. RNAfold is taken from $PATH unless RNAFOLD_BIN overrides it.
RNAFOLD_BIN = os.environ.get("RNAFOLD_BIN") or "RNAfold"


def fold_many(seqs: list[str], threads: int) -> list[float]:
    """RNAfold -p0 over all sequences in one process; MFE per nucleotide."""
    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as fh:
        for i, s in enumerate(seqs):
            fh.write(f">{i}\n{s.upper().replace('U', 'T')}\n")
        path = fh.name
    try:
        out = subprocess.run([RNAFOLD_BIN, "--noPS", "-j%d" % threads, "-i", path],
                             capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(path)
    vals, cur = [], None
    for line in out.splitlines():
        if line.startswith(">"):
            cur = None
        elif "(" in line and line.rstrip().endswith(")"):
            e = float(line[line.rfind("(") + 1:line.rfind(")")].strip())
            vals.append(e)
        elif cur is None and line.strip():
            cur = len(line.strip())
    lens = [len(s) for s in seqs]
    return [e / max(n, 1) for e, n in zip(vals, lens)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/real_gene_metrics.json")
    ap.add_argument("--fasta-dir", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--no-mfe", action="store_true",
                    help="skip RNAfold (fast path for the codon-only descriptors)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cm = _codon_metrics()
    rng = np.random.default_rng(args.seed)
    result = {}
    for task, (csv_path, lgbm) in TASKS.items():
        if not os.path.exists(csv_path):
            print(f"  ! missing {csv_path}"); continue
        high, low = load_split(csv_path)
        pools = {"real_low": low, "real_high": high}
        ref = cm.load_cai_reference(os.path.join(lgbm, "cai_reference.json"))
        tw = cm.load_tai_weights()

        per = {}
        for arm, pool in pools.items():
            seqs = [r["Sequence"] for r in pool]
            cl = [codons(s) for s in seqs]
            d = {"cai": np.array([cm.cai(c, ref) for c in cl]),
                 "tai": np.array([cm.tai(c, tw) for c in cl]),
                 "gc": np.array([cm.gc_content(c) for c in cl]),
                 "gc3": np.array([cm.gc3_content(c) for c in cl])}
            if not args.no_mfe:
                d["mfe_per_nt"] = np.array(fold_many(seqs, args.threads))
            per[arm] = d
            if args.fasta_dir:
                os.makedirs(args.fasta_dir, exist_ok=True)
                with open(os.path.join(args.fasta_dir, f"{task}_{arm}.fasta"), "w") as fh:
                    for r in pool:
                        fh.write(f">{r['seq_id']}\n"
                                 f"{r['Sequence'].upper().replace('U', 'T')}\n")
            print(f"  {task}/{arm}: n={len(pool)}")

        block = {}
        for arm in pools:
            block[arm] = {"n": len(pools[arm])}
            block[arm].update({m: float(per[arm][m].mean()) for m in per[arm]})
        # bootstrap the high-low difference over genes
        gaps = {}
        for m in per["real_high"]:
            hi, lo = per["real_high"][m], per["real_low"][m]
            bs = [hi[rng.integers(0, len(hi), len(hi))].mean()
                  - lo[rng.integers(0, len(lo), len(lo))].mean()
                  for _ in range(args.boot)]
            gaps[m] = {"gap": float(hi.mean() - lo.mean()),
                       "ci95": [float(np.percentile(bs, 2.5)),
                                float(np.percentile(bs, 97.5))],
                       "separates": bool(np.percentile(bs, 2.5) > 0
                                         or np.percentile(bs, 97.5) < 0)}
        block["_gap"] = gaps
        block["_provenance"] = {"test_csv": csv_path, "cai_reference": lgbm,
                                "rule": "entire top/bottom quartile of the TEST split",
                                "bootstrap": args.boot}
        result[task] = block

    if args.no_mfe and os.path.exists(args.out):     # keep any MFE already computed
        old = json.load(open(args.out))
        for t in result:
            for arm in ("real_high", "real_low"):
                if "mfe_per_nt" in old.get(t, {}).get(arm, {}):
                    result[t][arm]["mfe_per_nt"] = old[t][arm]["mfe_per_nt"]
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
