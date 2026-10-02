"""UTR-swap test.

The question is whether the model predicts reduced expression when an
otherwise codon-optimal CDS is placed between characterized destabilizing yeast
UTRs (PIR1, RNR1, ERG5, HTB1, HHF1). Each CDS is scored under every context, so
CDS identity is held fixed and only the UTR pair changes.

Two CDS sets are scored:
  - the fine-tuned model's designs for the target panel (the "otherwise
    codon-optimal CDS" the question is about)
  - the native CDS of the nine yeast genes themselves, which gives a
    within-yeast reference: if the scorer cannot separate contexts even for
    native transcripts, it cannot answer the question for designed ones.

Scored with the human translation-rate oracle (mRNA-LM fine-tuned on
data/translation_rate.csv) -- the only evaluator available here that was trained
on variable UTRs. The fungal evaluator saw one constant UTR context in training
and is structurally incapable of responding to UTR changes.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sft.paths import RUNS, external                             # noqa: E402

# mRNA-LM is third-party software and is not included in this repository: clone
# it yourself (https://github.com/sunjinyuan/mRNA-LM) and set MRNA_LM_REPO to the
# checkout -- or put it at <MRNA_GPT_EXTERNAL>/mRNA-LM.
MRNA_LM = str(external("mRNA-LM", "MRNA_LM_REPO"))
sys.path.insert(0, MRNA_LM)


def read_fasta(path: str) -> dict[str, str]:
    seqs, cur_id, cur = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(cur)
                cur_id, cur = line[1:].split()[0], []
            else:
                cur.append(line)
        if cur_id is not None:
            seqs[cur_id] = "".join(cur)
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", required=True, metavar="LABEL=PATH", action="append")
    ap.add_argument("--contexts", default="sft/data/yeast_utr_contexts.json")
    ap.add_argument("--ckpt", default=str(RUNS / "mrna_lm_tr" / "best_model.pt"))
    ap.add_argument("--per-target", type=int, default=50,
                    help="subsample this many designs per target (0 = all)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md-out", required=True)
    args = ap.parse_args()

    import torch
    from scipy.stats import mannwhitneyu

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_mrna_lm_score", os.path.join(REPO, "evaluate", "mrna_lm_score.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ctx = json.load(open(args.contexts))
    order = [g for g in ctx if ctx[g]["group"] == "destabilizing"] + \
            [g for g in ctx if ctx[g]["group"] == "stable"]
    print("contexts: " + ", ".join(f"{g}({ctx[g]['group'][:4]})" for g in order), flush=True)

    cds_sets: dict[str, list[str]] = {}
    for spec_s in args.designs:
        label, _, path = spec_s.partition("=")
        seqs = read_fasta(path)
        by_target: dict[str, list[str]] = {}
        for sid, s in seqs.items():
            by_target.setdefault(sid.split("|")[1], []).append(s)
        picked = []
        for t, lst in sorted(by_target.items()):
            picked.extend(lst[:args.per_target] if args.per_target else lst)
        cds_sets[label] = picked
        print(f"{label}: {len(picked)} CDS from {len(by_target)} targets", flush=True)
    cds_sets["native_yeast_cds"] = [ctx[g]["cds"] for g in order]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = mod.load_scorer(args.ckpt, device=device, num_labels=1)

    results: dict = {}
    for label, seqs in cds_sets.items():
        results[label] = {}
        for g in order:
            res = mod.score_batch(model, seqs, ctx[g]["utr5"], ctx[g]["utr3"],
                                  device=device, batch_size=args.batch_size)
            vals = [r["score"] for r in res]
            results[label][g] = vals
            print(f"{label} @ {g} ({ctx[g]['group']}): n={len(vals)} "
                  f"mean {statistics.mean(vals):+.4f}", flush=True)

    summary: dict = {"contexts": {g: {"group": ctx[g]["group"], "gene_id": ctx[g]["gene_id"]}
                                  for g in order}, "sets": {}}
    md = ["# UTR-swap test", "",
          "The same CDS is scored under 9 fixed yeast UTR contexts: the CDS is held "
          "constant and only the UTR pair changes. The scorer is the human "
          "translation-rate oracle (mRNA-LM fine-tuned on data/translation_rate.csv, "
          "held-out-fold Pearson 0.617) -- the only evaluator here that was trained "
          "on variable UTRs.", "",
          "The UTRs are 200 nt genomic-flank proxies (yeast UTRs are unannotated); "
          "the ADH1 flanks match the independently retrieved real NCBI UTRs "
          "base for base, validating the extraction.", "",
          "| CDS set | " + " | ".join(f"{g}<br>{ctx[g]['group'][:4]}" for g in order) +
          " | destabilizing mean | stable mean | Δ | Mann-Whitney p |",
          "|---|" + "---:|" * (len(order) + 4)]
    for label, per_ctx in results.items():
        des = [v for g in order if ctx[g]["group"] == "destabilizing" for v in per_ctx[g]]
        sta = [v for g in order if ctx[g]["group"] == "stable" for v in per_ctx[g]]
        try:
            p = mannwhitneyu(des, sta, alternative="two-sided")[1]
        except ValueError:
            p = float("nan")
        summary["sets"][label] = {
            "per_context_mean": {g: statistics.mean(per_ctx[g]) for g in order},
            "destabilizing_mean": statistics.mean(des), "stable_mean": statistics.mean(sta),
            "delta": statistics.mean(des) - statistics.mean(sta),
            "mannwhitney_p": float(p), "n_per_context": len(per_ctx[order[0]])}
        cells = " | ".join(f"{statistics.mean(per_ctx[g]):+.3f}" for g in order)
        s = summary["sets"][label]
        md.append(f"| {label} | {cells} | {s['destabilizing_mean']:+.3f} | "
                  f"{s['stable_mean']:+.3f} | {s['delta']:+.3f} | {p:.2e} |")
    md += ["", "Δ = mean over destabilizing contexts − mean over stable contexts. "
           "A negative value means the scorer predicts lower values under the "
           "destabilizing UTRs.", ""]

    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(args.md_out, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"wrote {args.out} and {args.md_out}")


if __name__ == "__main__":
    main()
