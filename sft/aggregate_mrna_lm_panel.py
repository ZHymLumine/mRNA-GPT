"""Aggregate mRNA-LM 5-class scores over the target panel, per target protein.

evaluate/mrna_lm_score.py emits one JSON line per generated sequence; this
groups those by the target encoded in the FASTA id ("{label}|{target}|{i}") and
reports, per target and per checkpoint, the mean class-probability vector and
the argmax-class distribution.

The published task's five classes are HPA-style expression-*specificity*
classes, not a low->high ordinal, so no single scalar is reported: the honest
statement is how the distribution shifts, plus the fact that both batches were
scored under the identical ADH1 UTR context by a model never fine-tuned on our
fungal data.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter

N_CLASS = 5


def load(path: str) -> dict[str, list[list[float]]]:
    groups: dict[str, list[list[float]]] = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            parts = rec["id"].split()[0].split("|")   # drop " n_codon=..." then split
            if len(parts) < 3:
                raise ValueError(f"id {rec['id']!r} carries no target field")
            groups.setdefault(parts[1], []).append(rec["probs"])
    return groups


def summarize(rows: list[list[float]]) -> dict:
    return {
        "n": len(rows),
        "mean_probs": [statistics.mean(r[c] for r in rows) for c in range(N_CLASS)],
        "argmax_distribution": {
            str(c): Counter(max(range(N_CLASS), key=r.__getitem__) for r in rows).get(c, 0)
            for c in range(N_CLASS)
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained-jsonl", required=True)
    ap.add_argument("--sft-jsonl", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    labels = ("pretrained_eukaryote", "fungal_sft")
    data = {labels[0]: load(args.pretrained_jsonl), labels[1]: load(args.sft_jsonl)}
    targets = [t for t in data[labels[0]] if t in data[labels[1]]]
    missing = (set(data[labels[0]]) | set(data[labels[1]])) - set(targets)
    if missing:
        raise SystemExit(f"targets scored for only one checkpoint: {sorted(missing)}")

    out = {lab: {t: summarize(data[lab][t]) for t in targets} for lab in labels}
    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=2)

    md = ["## mRNA-LM 5-class expression-specificity scores (fixed ADH1 UTR context)", "",
          "The evaluator is the published mRNA-LM, LoRA fine-tuned once on its own "
          "protein_expression_5class task (test AUROC 0.689 / F1 0.314) and never "
          "exposed to this work's fungal data. The five classes are HPA-style "
          "expression *specificity* classes, not a low->high ordinal, so the "
          "distribution is reported rather than a single score.", ""]
    for lab in labels:
        md += [f"### {lab}", "",
               "| target protein | n | " + " | ".join(f"P(class {c})" for c in range(N_CLASS)) + " | argmax distribution |",
               "|---|---:|" + "---:|" * N_CLASS + "---|"]
        for t in targets:
            s = out[lab][t]
            probs = " | ".join(f"{p:.3f}" for p in s["mean_probs"])
            dist = ", ".join(f"{c}:{n}" for c, n in s["argmax_distribution"].items() if n)
            md.append(f"| {t} | {s['n']} | {probs} | {dist} |")
        md.append("")
    with open(args.out_md, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
