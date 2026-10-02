"""Score every design method's panel sequences with the mRNA-LM TRANSLATION-RATE
oracle -- the primary CDS readout in Li et al., mRNA-GPT, which benchmarks the
same four target proteins.

Two things this does that the paper's setup does not:

  - Reports under TWO fixed UTR contexts (yeast ADH1 and a human transcript from
    the oracle's own training distribution). The oracle is trained on human
    transcripts, so a fungal UTR context is off-distribution for it; running both
    shows whether the ranking is an artifact of that choice. Every method sees
    the identical context, so the comparison is internally fair either way.
  - The oracle is NOT our optimization target. Li et al. fine-tune directly on
    mRNA-LM scores for 10 iterations and then report mRNA-LM scores, with
    zero-shot baselines; our model never saw this oracle, so a win here would
    mean something different from a win there (and a loss is equally meaningful).

Mann-Whitney U (the test used in that paper) compares each method against
mrna_gpt_sft per target, two-sided, on the raw per-sequence scores.
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
GEMORNA_LM = str(external("mRNA-LM", "MRNA_LM_REPO"))
sys.path.insert(0, GEMORNA_LM)


def read_fasta(path: str) -> dict[str, str]:
    """Inlined rather than imported from sft.compare_pretrained_vs_sft: that
    module pulls in evaluate.* and lightgbm at import time, neither of which is
    available (or needed) in the mrna_lm env."""
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


def group_by_target(seqs: dict[str, str]) -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    for sid, seq in seqs.items():
        parts = sid.split("|")
        if len(parts) < 3:
            raise ValueError(f"id {sid!r} is not '{{label}}|{{target}}|{{i}}'")
        groups.setdefault(parts[1], {})[sid] = seq
    return groups


def pick_human_utr(path: str, fold: int = 5) -> tuple[str, str, str]:
    """A single fixed human UTR pair from the oracle's own test fold: the
    transcript whose 5' and 3' UTR lengths are closest to the dataset medians.
    Fixed for every method and reported by ID so the choice is auditable."""
    import pandas as pd
    df = pd.read_csv(path).fillna("")
    df = df[df["split"] == fold]
    m5, m3 = df.UTR5.astype(str).map(len).median(), df.UTR3.astype(str).map(len).median()
    d = ((df.UTR5.astype(str).map(len) - m5).abs() / max(m5, 1)
         + (df.UTR3.astype(str).map(len) - m3).abs() / max(m3, 1))
    row = df.loc[d.idxmin()]
    return str(row.UTR5), str(row.UTR3), str(row.ENSTID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--ckpt", default=str(RUNS / "mrna_lm_tr" / "best_model.pt"))
    ap.add_argument("--tr-csv", default=os.path.join(GEMORNA_LM, "data/translation_rate.csv"))
    ap.add_argument("--reference", default="mrna_gpt_sft", help="method the U tests compare against")
    ap.add_argument("--evaluator-note", default=None,
                    help="one-line provenance of the evaluator, written into the report. "
                         "Defaults to the human translation-rate description; ALWAYS pass this "
                         "when scoring with a different checkpoint, or the report will misstate "
                         "what produced the numbers.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--contexts", default="adh1_yeast,human_median",
                    # NOTE: the context MUST match the one the evaluator was trained
                    # under. Each property evaluator holds a single context constant
                    # during fine-tuning, so scoring under any other UTR pair is
                    # off-distribution and the numbers are meaningless.
                    help="comma-separated UTR contexts to score under. The fungal "
                         "evaluator was trained with the ADH1 context held constant, so "
                         "scoring it under a human context is off-distribution and "
                         "meaningless -- pass adh1_yeast alone for that model.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md-out", required=True)
    args = ap.parse_args()

    import torch
    from scipy.stats import mannwhitneyu

    # Load evaluate/mrna_lm_score.py by path: the mrna_lm env has HuggingFace's
    # "evaluate" package installed, and an installed regular package shadows our
    # namespace-package directory of the same name whatever sys.path says.
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "_mrna_lm_score", os.path.join(REPO, "evaluate", "mrna_lm_score.py"))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    load_scorer, score_batch = _mod.load_scorer, _mod.score_batch
    from sft.data.adh1_utr_context import ADH1_5UTR, ADH1_3UTR
    from sft.prepare_property_mrna_lm_data import ECOLI_3UTR, ECOLI_5UTR

    h5, h3, hid = pick_human_utr(args.tr_csv)
    contexts = {"adh1_yeast": (ADH1_5UTR, ADH1_3UTR, "yeast ADH1"),
                "human_median": (h5, h3, f"human {hid}"),
                "ecoli_rbs": (ECOLI_5UTR, ECOLI_3UTR,
                              "E. coli consensus SD / terminator")}
    want = [c.strip() for c in args.contexts.split(",") if c.strip()]
    unknown = [c for c in want if c not in contexts]
    if unknown:
        raise SystemExit(f"unknown context(s) {unknown}; choose from {list(contexts)}")
    contexts = {c: contexts[c] for c in want}
    print("UTR contexts in use: " + ", ".join(
        f"{c} ({len(contexts[c][0])}/{len(contexts[c][1])} nt)" for c in contexts), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_scorer(args.ckpt, device=device, num_labels=1)

    methods = [s.partition("=")[0::2] for s in args.fasta]
    raw: dict = {}
    for label, path in methods:
        if not os.path.exists(path):
            raise SystemExit(f"{label}: no such FASTA {path}")
        raw[label] = {}
        for target, sub in group_by_target(read_fasta(path)).items():
            seqs = list(sub.values())
            raw[label][target] = {}
            for cname, (u5, u3, _) in contexts.items():
                res = score_batch(model, seqs, u5, u3, device=device, batch_size=args.batch_size)
                raw[label][target][cname] = [r["score"] for r in res]
            print(f"{label}/{target}: n={len(seqs)} "
                  + " ".join(f"{c}={statistics.mean(raw[label][target][c]):+.3f}"
                             for c in contexts), flush=True)

    summary: dict = {"utr_contexts": {c: {"desc": contexts[c][2]} for c in contexts},
                     "oracle_ckpt": args.ckpt, "methods": {}}
    for label, _ in methods:
        summary["methods"][label] = {}
        for target, per_ctx in raw[label].items():
            entry = {}
            for cname, vals in per_ctx.items():
                e = {"n": len(vals), "mean": statistics.mean(vals),
                     "median": statistics.median(vals),
                     "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0}
                ref = raw.get(args.reference, {}).get(target, {}).get(cname)
                if ref and label != args.reference and len(vals) > 0 and len(ref) > 0:
                    try:
                        u, p = mannwhitneyu(vals, ref, alternative="two-sided")
                        e["mannwhitney_u_vs_ref"] = float(u)
                        e["mannwhitney_p_vs_ref"] = float(p)
                    except ValueError as exc:      # identical constant samples
                        e["mannwhitney_note"] = str(exc)
                e["scores"] = vals
                # A deterministic method emits ONE sequence, so a U test against a
                # 50-sample generator has essentially no power (that is why the
                # table's asterisks are unreliable for them). The question that IS
                # answerable: where does that single design fall inside the
                # generator's own distribution, and does the generator's best-of-n
                # beat it -- which is how a sampler would actually be used.
                if ref and len(vals) == 1 and len(ref) > 1:
                    e["percentile_in_ref"] = 100.0 * sum(r < vals[0] for r in ref) / len(ref)
                    e["ref_best_of_n"] = max(ref)
                    e["beaten_by_ref_best"] = max(ref) > vals[0]
                entry[cname] = e
            summary["methods"][label][target] = entry
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)

    targets = sorted({t for label, _ in methods for t in raw[label]})
    note = args.evaluator_note or (
        "The evaluator is mRNA-LM, LoRA fine-tuned on its own translation_rate task "
        "(human transcripts; held-out-fold test Pearson 0.617 / Spearman 0.614). It was "
        "never trained on any data from this work and is not an optimization target "
        "here. ")
    md = [f"# mRNA-LM scores (evaluator checkpoint: {os.path.basename(os.path.dirname(args.ckpt))})", "",
          note + "Every method uses exactly the same fixed UTR context.", "",
          f"UTR contexts: `adh1_yeast` = yeast ADH1; `human_median` = {hid} "
          f"(a human transcript inside this oracle's training distribution, of close "
          f"to median length).", ""]
    for cname in contexts:
        md += [f"## UTR context: {contexts[cname][2]}", "",
               "| method | " + " | ".join(targets) + " |", "|---|" + "---:|" * len(targets)]
        for label, _ in methods:
            cells = []
            for t in targets:
                e = summary["methods"][label].get(t, {}).get(cname)
                if not e:
                    cells.append("–")
                    continue
                cell = f"{e['mean']:+.3f}"
                p = e.get("mannwhitney_p_vs_ref")
                if p is not None:
                    cell += "\\*" if p < 0.05 else " (ns)"
                cells.append(cell)
            md.append(f"| {label} | " + " | ".join(cells) + " |")
        md += ["", f"\\* two-sided Mann-Whitney U against `{args.reference}`, p<0.05; "
               "ns = not significant. "
               "**A method with n=1 has a single deterministic output, so the U test has "
               "almost no power and its significance marks cannot be trusted -- read "
               "the percentiles in the next section instead.**", ""]

        det = [lab for lab, _ in methods
               if summary["methods"][lab].get(targets[0], {}).get(cname, {}).get("n") == 1]
        if det:
            md += [f"### Where deterministic methods (n=1) fall in the `{args.reference}` distribution", "",
                   f"A single design against a sampler with n={len(raw[args.reference][targets[0]][cname])}: "
                   "the answerable question is not \"whose mean is higher\" but \"at which "
                   "percentile of the sampler's distribution does this design sit\", and "
                   "\"can the sampler's best-of-n beat it\" -- the latter being how a "
                   "sampler is actually used.", "",
                   "| method | " + " | ".join(f"{t} percentile" for t in targets) + " | beaten by best-of-n |",
                   "|---|" + "---:|" * len(targets) + "---|"]
            for lab in det:
                cells, beaten = [], 0
                for t in targets:
                    e = summary["methods"][lab].get(t, {}).get(cname, {})
                    pc = e.get("percentile_in_ref")
                    cells.append("–" if pc is None else f"P{pc:.0f}")
                    beaten += bool(e.get("beaten_by_ref_best"))
                md.append(f"| {lab} | " + " | ".join(cells) + f" | {beaten}/{len(targets)} |")
            ref_line = " | ".join(
                f"{max(raw[args.reference][t][cname]):+.3f}" for t in targets)
            md += ["", f"best-of-n for `{args.reference}`: " + ref_line, ""]
    with open(args.md_out, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"wrote {args.out} and {args.md_out}")


if __name__ == "__main__":
    main()
