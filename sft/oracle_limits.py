#!/usr/bin/env python3
"""Audit a neural evaluator before its ranking is quoted.

The fungal report carries this analysis hand-written; this makes it a script so
the stability evaluator gets the identical treatment and either can be redone.

Two failure modes are checked, both of which make a neural oracle silently
endorse CAI-maximizing methods:

1. LEARNED CAI CORRELATION. The evaluator takes no CAI input feature, but real
   genes' codon adaptation does correlate with the measured property, so any
   model fit to real genes learns some of it. Reported as the correlation
   between the oracle's per-cell mean and that cell's CAI.

2. OUT-OF-DISTRIBUTION EXTRAPOLATION. A method whose CAI sits above the
   evaluator's own training range is being scored in a region no real gene
   supports. cai_max (CAI ~1.0) and LinearDesign at high lambda land there by
   construction, so a high score for them means nothing.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.codon_metrics import cai, load_cai_reference   # noqa: E402


def to_codons(seq: str) -> list[str]:
    seq = seq.upper().replace("T", "U")
    return [seq[i:i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def pct(sorted_vals, q: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="JSON from score_panel_translation_rate.py")
    ap.add_argument("--realism", required=True,
                    help="JSON from evaluate_host_and_realism.py (supplies per-cell CAI)")
    ap.add_argument("--lm-csv", required=True,
                    help="the evaluator's own training CSV (UTR5,CDS,UTR3,y,split)")
    ap.add_argument("--cai-ref", required=True)
    ap.add_argument("--context", default=None,
                    help="which UTR context in --scores to audit (default: the only one)")
    ap.add_argument("--cai-key", default="cai_ref", help="CAI field in the realism JSON")
    ap.add_argument("--md-out", required=True)
    args = ap.parse_args()

    from scipy.stats import pearsonr, spearmanr

    sc = json.load(open(args.scores))
    rl = json.load(open(args.realism))
    ctxs = list(sc.get("utr_contexts", {})) or ["default"]
    ctx = args.context or ctxs[0]

    # ---- per-cell (method, target) oracle mean vs CAI ----
    xs, ys, cells = [], [], []
    for m, per_target in sc.get("methods", {}).items():
        for t, per_ctx in per_target.items():
            e = per_ctx.get(ctx) if isinstance(per_ctx, dict) else None
            if not e or "mean" not in e:
                continue
            c = rl.get(m, {}).get(t, {}).get(args.cai_key)
            if c is None:
                continue
            xs.append(c); ys.append(e["mean"]); cells.append((m, t))
    if len(xs) < 3:
        raise SystemExit(f"only {len(xs)} comparable cells; check --context/--cai-key")
    pr = pearsonr(xs, ys)[0]
    sr = spearmanr(xs, ys)[0]
    print(f"{len(xs)} cells: oracle vs CAI  Pearson {pr:+.3f}  Spearman {sr:+.3f}")

    # ---- the evaluator's own training CAI distribution ----
    csv.field_size_limit(10 ** 9)
    w = load_cai_reference(args.cai_ref)
    train_cai = sorted(cai(to_codons(r["CDS"]), w)
                       for r in csv.DictReader(open(args.lm_csv)) if r.get("CDS"))
    q = {k: pct(train_cai, v) for k, v in
         (("min", 0.0), ("P50", 0.5), ("P99", 0.99), ("max", 1.0))}
    print(f"training CAI: min {q['min']:.3f} / P50 {q['P50']:.3f} / "
          f"P99 {q['P99']:.3f} / max {q['max']:.3f}")

    per_method: dict[str, list[float]] = {}
    for (m, _t), c in zip(cells, xs):
        per_method.setdefault(m, []).append(c)
    rows = []
    for m, cs in sorted(per_method.items(), key=lambda kv: -max(kv[1])):
        lo, hi = min(cs), max(cs)
        if lo > q["max"]:
            verdict = ("**entirely above the training-set maximum -> "
                       "out-of-distribution extrapolation, the score is meaningless**")
        elif hi > q["max"]:
            verdict = "**partly above the training-set maximum -> out of distribution**"
        elif hi > q["P99"]:
            verdict = "close to the training-set upper bound (>P99) -> treat with caution"
        else:
            verdict = "inside the training distribution"
        rows.append((m, lo, hi, verdict))

    md = [f"# Limits of the neural evaluator itself ({os.path.basename(args.scores)})", "",
          f"UTR context: `{ctx}`. Both points below must be read together with the "
          f"score table.", "",
          "## 1. It has no CAI input feature, yet it still learned CAI", "",
          f"Over the {len(xs)} method x target-protein cells, the correlation between "
          f"the evaluator mean and that cell's CAI is "
          f"**Pearson {pr:+.3f} / Spearman {sr:+.3f}**.", "",
          "In real genes CAI is already correlated with this property, so any model "
          "fitted to real genes picks that up. It is therefore cleaner than LightGBM, "
          "but **not independent of CAI**.", "",
          "## 2. Methods outside the training distribution cannot be ranked with it", "",
          f"CAI distribution of the evaluator's own training data: min {q['min']:.3f} / "
          f"P50 {q['P50']:.3f} / "
          f"**P99 {q['P99']:.3f} / max {q['max']:.3f}** (n={len(train_cai):,}).", "",
          "| method | CAI range | verdict |", "|---|---|---|"]
    for m, lo, hi, verdict in rows:
        md.append(f"| {m} | {lo:.3f} – {hi:.3f} | {verdict} |")
    md += ["", "## Conclusion", "",
           "This table may be used to compare **methods that are inside the "
           "distribution**; the high scores it gives to methods outside the training "
           "CAI range (the CAI-maximizing ones) are extrapolation, with no real gene "
           "anywhere in that region to support them. Those methods can only be judged "
           "by the JS/LLR metrics anchored on real measured values -- and under those "
           "metrics their JS is tens to hundreds of times further from the real-gene "
           "profile.", ""]
    os.makedirs(os.path.dirname(os.path.abspath(args.md_out)), exist_ok=True)
    with open(args.md_out, "w") as fh:
        fh.write("\n".join(md))
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()
