#!/usr/bin/env python3
"""Read out the bacterial-expression learning-rate sweep.

The number that decides this sweep is the SEPARATION between the high arm and
the low control, not either arm's own score. The observed failure at the
original learning rate was that both arms moved the same distance in the same
direction -- a setting that lifts both equally has not produced evidence, only a
better-looking number.

Separation is reported in units of the real high-minus-low difference, so 1.0
means the two fine-tuned arms are as far apart as real high- and low-expression
genes are, and 0.0 means they are indistinguishable.

Selection happens on VAL and is reported on TEST. Picking the learning rate on
the same held-out genes the result is quoted against is the circularity this
paper objects to elsewhere; doing it here would be no better.

    python sft/summarize_bacexp_sweep.py --sweep <dir> --out reports/...md
"""
from __future__ import annotations

import argparse
import csv
import json
import os

LRS = ["1e5", "3e5", "1e4", "3e4"]
PRETTY = {"1e5": "1e-5", "3e5": "3e-5 (original)", "1e4": "1e-4", "3e4": "3e-4"}
KEYS = [("llr_high_low", "codon LLR"), ("js_high_minus_low", "JS(high)-JS(low)")]


def val_loss(sweep: str, arm: str, lr: str):
    p = os.path.join(sweep, f"{arm}_lr{lr}", "metrics.csv")
    if not os.path.exists(p):
        return None, None
    ev = [r for r in csv.DictReader(open(p)) if r.get("val_loss")]
    if not ev:
        return None, None
    b = min(ev, key=lambda r: float(r["val_loss"]))
    return float(b["val_loss"]), b["global_step"]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def arm_val(d, label, key):
    """Realism jsons key generated arms under 'all', real rows under '_all'."""
    if not d or label not in d:
        return None
    blk = d[label]
    sub = blk.get("all", blk.get("_all"))
    return None if sub is None else sub.get(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = {s: load(os.path.join(args.sweep, f"sweep_realism_{s}.json"))
            for s in ("val", "test")}

    md = ["# Bacterial protein expression: learning-rate sweep", "",
          "What is swept is the **separation between the high-property arm and the "
          "low-property control**, not the score of either arm on its own.",
          "The failure mode of the original setting is that both arms move the same "
          "distance in the same direction; a setting that lifts both together "
          "provides no new evidence.", "",
          "Separation is expressed in units of the **real high-minus-low gene "
          "difference**: 1.0 means the distance between the two fine-tuned arms "
          "equals the distance between real high- and low-expression genes, and 0.0 "
          "means they are indistinguishable.", "",
          "**Selection on VAL, reporting on TEST.** Choosing hyperparameters on the "
          "same held-out genes the result is quoted from is exactly the circularity "
          "this work argues against elsewhere.", ""]

    for key, name in KEYS:
        md += [f"## Separation: {name}", "",
               "| learning rate | VAL separation | TEST separation | high arm val loss | low arm val loss |",
               "|---|---:|---:|---:|---:|"]
        rows = []
        for lr in LRS:
            cell = {}
            for split in ("val", "test"):
                d = data[split]
                hi = arm_val(d, f"high_lr{lr}", key)
                lo = arm_val(d, f"low_lr{lr}", key)
                rh = arm_val(d, "REAL_high_test", key)
                rl = arm_val(d, "REAL_low_test", key)
                if None in (hi, lo, rh, rl) or rh == rl:
                    cell[split] = None
                else:
                    cell[split] = (hi - lo) / (rh - rl)
            vh, _ = val_loss(args.sweep, "high", lr)
            vl, _ = val_loss(args.sweep, "low", lr)
            rows.append((lr, cell, vh, vl))
            f = lambda x, n=3: "–" if x is None else f"{x:.{n}f}"   # noqa: E731
            md.append(f"| {PRETTY[lr]} | {f(cell['val'])} | {f(cell['test'])} | "
                      f"{f(vh, 4)} | {f(vl, 4)} |")
        md.append("")
        picked = [r for r in rows if r[1]["val"] is not None]
        if picked:
            best = max(picked, key=lambda r: r[1]["val"])
            best_test = best[1]["test"]
            shown = "–" if best_test is None else f"{best_test:.3f}"
            md += [f"**Selected on VAL: lr = {PRETTY[best[0]]}**"
                   f" (VAL separation {best[1]['val']:.3f}); "
                   f"its TEST separation is {shown}.", ""]

    md += ["## Dimensions that were not swept, and why", "",
           "- **Training steps**: both arms reach their best val loss at epoch 3 of 60 "
           "and get monotonically worse afterwards. That is early stopping, not "
           "under-training; more steps would only deepen the overfitting.",
           "- **A hidden problem in the original schedule**: the original config paired "
           "`max_epochs: 60` with cosine decay but actually stopped at epoch 3, so the "
           "learning rate never really decayed. This sweep uses `max_epochs: 12` "
           "throughout so the cosine schedule completes, which is why 3e-5 was "
           "retrained here too and the original run cannot simply be reused.", ""]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write("\n".join(md) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
