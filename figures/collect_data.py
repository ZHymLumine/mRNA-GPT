"""Gather every number the result figures plot into one JSON.

Keeping collection separate from plotting means the figures are reproducible
from a single file, and any number in the manuscript can be traced back to the
run directory it came from.
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sft.paths import DATA as _DATA, RUNS as _RUNS              # noqa: E402

RUNS = str(_RUNS)
DATA = str(_DATA)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure_data.json")


def load(p):
    if not os.path.exists(p):
        print(f"  missing: {p}")
        return None
    return json.load(open(p))


def to_codons(s):
    s = s.upper().replace("T", "U")
    return [s[i:i + 3] for i in range(0, len(s) - len(s) % 3, 3)]


def read_fasta(p):
    seqs, cur = {}, []
    sid = None
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if sid:
                seqs[sid] = "".join(cur)
            sid, cur = line[1:].split()[0], []
        else:
            cur.append(line)
    if sid:
        seqs[sid] = "".join(cur)
    return seqs


def main():
    out: dict = {}

    # ---------- Fig 2: pretraining ----------
    dom = {}
    for d in ("archaea", "bacteria", "eukaryote"):
        ev = load(f"reports/{d}_final_eval.json")
        dom[d] = {"val_loss": ev[0]["loss"], "val_ppl": ev[0]["ppl"],
                  "val_seqs": ev[0]["seqs"]} if ev else {}
    # leakage, transcribed from reports/leakage_comparison.md (mmseqs measurement)
    out["leakage"] = {
        "archaea":   {"random": [80.46, 78.55, 33.16, 2.51], "ours": [0.52, 0.28, 0.00, 0.00]},
        "bacteria":  {"random": [81.64, 79.36, 34.34, 3.33], "ours": [0.01, 0.00, 0.00, 0.00]},
        "eukaryote": {"random": [72.38, 71.82, 56.60, 28.69], "ours": [0.03, 0.03, 0.00, 0.00]},
        "thresholds": [">=50%", ">=70%", ">=90%", ">=99%"]}
    # de novo CDS validity vs the training-set baseline. Parsed, not transcribed:
    # hand-copying these got two of three wrong on the first attempt.
    import re

    def _valid_pct(path, row_prefix):
        for line in open(path):
            if line.startswith(row_prefix):
                m = re.findall(r"\*\*([\d.]+)%\*\*", line)
                if m:
                    return float(m[-1])
        raise SystemExit(f"no validity row {row_prefix!r} in {path}")

    out["cds_validity"] = {
        d: {"train": _valid_pct(f"reports/{d}_qc_stats.md", "| all |"),
            "generated": _valid_pct(f"{RUNS}/{d}/unconstrained_qc.md", "| unconstrained |")}
        for d in ("archaea", "bacteria", "eukaryote")}
    xd = {}
    for m in ("archaea", "bacteria", "eukaryote"):
        r = load(f"{RUNS}/_crossdomain/model_{m}.json")
        if r:
            xd[m] = {row["label"]: row["ppl"] for row in r}
    out["cross_domain_ppl"] = xd
    out["domains"] = dom

    # ---------- Fig 3: de novo, two properties ----------
    denovo = {}
    s = load(f"{RUNS}/stability_sft/generation/stability_realism_unconstrained.json")
    if s:
        denovo["stability"] = {
            "arms": {k: s[k]["all"] for k in
                     ("pretrained_archaea", "stability_sft", "stability_sft_LOW") if k in s},
            "real_high": s["REAL_high_test"]["_all"], "real_low": s["REAL_low_test"]["_all"]}
    f = load(f"{RUNS}/fungal_sft/generation_low/unconstrained_negctrl.json")
    if f:
        denovo["expression"] = {
            "arms": {k: f[k]["all"] for k in
                     ("uncon_pretrained", "uncon_sft_HIGH", "uncon_sft_LOW") if k in f},
            "real_high": f["REAL_high_expression_test"]["_all"],
            "real_low": f["REAL_low_expression_test"]["_all"]}
    # Bacterial expression: the pretrained arm is unchanged, but the two
    # fine-tuned arms come from runs/bacexp_sweep, not runs/bacexp_sft -- the
    # original schedule's cosine never decayed and its low-property control
    # failed (see configs/sweep/). Both files score against the same real
    # high/low TEST genes, so the arms are directly comparable.
    b0 = load(f"{RUNS}/bacexp_sft/generation/bacexp_realism_unconstrained.json")
    bs = load(f"{RUNS}/bacexp_sweep/sweep_realism_test.json")
    if b0 and bs:
        denovo["bacexp"] = {
            "arms": {"mrna_gpt_pretrained": b0["mrna_gpt_pretrained"]["all"],
                     "high_lr1e5": bs["high_lr1e5"]["all"],
                     "low_lr1e5": bs["low_lr1e5"]["all"]},
            "real_high": bs["REAL_high_test"]["_all"],
            "real_low": bs["REAL_low_test"]["_all"]}
    out["denovo"] = denovo

    # independent tree predictor on the de novo batches
    from evaluate.lightgbm_expression import load_predictor, predict
    pred = {}
    specs = {
        "stability": ("sft/lightgbm_stability", "sft/data/mrna_stability_test.csv", {
            "pretrained": f"{RUNS}/stability_sft/generation/pretrained_archaea.fasta",
            "sft": f"{RUNS}/stability_sft/generation/stability_sft.fasta",
            "sft_low": f"{RUNS}/stability_sft/generation/stability_sft_low.fasta"}),
        "expression": ("sft/lightgbm_expression", "sft/data/fungal_expression_test.csv", {
            "pretrained": f"{RUNS}/fungal_sft/generation/pretrained_eukaryote.fasta",
            "sft": f"{RUNS}/fungal_sft/generation/fungal_sft.fasta",
            "sft_low": f"{RUNS}/fungal_sft/generation_low/fungal_sft_low.fasta"}),
        "bacexp": ("sft/lightgbm_bacexp", "sft/data/bacteria_expression_test.csv", {
            "pretrained": f"{RUNS}/bacexp_sft/generation/mrna_gpt_pretrained.fasta",
            "sft": f"{RUNS}/bacexp_sweep/high_lr1e5/denovo.fasta",
            "sft_low": f"{RUNS}/bacexp_sweep/low_lr1e5/denovo.fasta"})}
    for prop, (mdir, test_csv, fastas) in specs.items():
        m, ref = load_predictor(mdir)
        rows = list(csv.DictReader(open(test_csv)))
        rows.sort(key=lambda r: float(r["Value"]), reverse=True)
        k = int(len(rows) * 0.25)
        e = {"real_high": float(statistics.mean(predict([r["Sequence"] for r in rows[:k]], m, ref))),
             "real_low": float(statistics.mean(predict([r["Sequence"] for r in rows[-k:]], m, ref)))}
        for arm, path in fastas.items():
            if not os.path.exists(path):
                print(f"  missing fasta {path}")
                continue
            v = predict(list(read_fasta(path).values()), m, ref)
            e[arm] = {"mean": float(v.mean()), "sd": float(v.std()),
                      "values": [float(x) for x in v]}
        pred[prop] = e
    out["denovo_predictor"] = pred

    # ---------- Fig 4: protein-constrained panel ----------
    # All three panels at n=200 per target, so the tasks are directly comparable.
    panel = {}
    s = load(f"{RUNS}/stability_sft/generation_panel_n200/realism_per_target_n200.json")
    if s:
        panel["stability"] = {"data": s, "arms": ["mrna_gpt_pretrained", "mrna_gpt_sft",
                                                  "mrna_gpt_sft_LOW"],
                              "real": ["REAL_high_test", "REAL_low_test"]}
    f = load(f"{RUNS}/fungal_sft/generation_panel_n200/realism_final.json")
    if f:
        panel["expression"] = {"data": f, "arms": ["mrna_gpt_pretrained", "mrna_gpt_sft",
                                                   "mrna_gpt_sft_LOW"],
                               "real": ["REAL_high_expression_test", "REAL_low_expression_test"]}
    b = load(f"{RUNS}/bacexp_sweep/generation_panel_n200/realism_per_target.json")
    if b:
        panel["bacexp"] = {"data": b, "arms": ["mrna_gpt_pretrained", "mrna_gpt_sft",
                                               "mrna_gpt_sft_LOW"],
                           "real": ["REAL_high_test", "REAL_low_test"]}
    out["panel"] = panel
    out["panel_identity"] = panel_identity()

    # ---------- Fig 5: all methods ----------
    out["all_methods"] = {
        "stability_realism": stability_all_methods_n200(),
        # Neural evaluators fitted on the VAL+TEST splits only, so they have seen
        # neither the fine-tuning sequences nor their homologs. The higher-scoring
        # *_std checkpoints train on the split the fine-tuning set is drawn from
        # and are deliberately not used for any reported number.
        "stability_oracle": load(f"{RUNS}/stability_sft/generation_panel/"
                                 "stability_oracle_scores.json"),
        "expression_realism": load(f"{RUNS}/fungal_sft/generation_panel_n200/realism_final.json"),
        "expression_oracle": load(f"{RUNS}/fungal_sft/generation_panel_n200/"
                                  "fungal_oracle_all_methods.json"),
        "stability_property_eval": load(f"{RUNS}/stability_sft/generation_panel/"
                                        "all_methods_property_eval.json"),
        "expression_property_eval": load(f"{RUNS}/fungal_sft/generation_panel/"
                                         "all_methods_property_eval.json"),
        "bacexp_realism": load(f"{RUNS}/bacexp_sweep/generation_panel_n200/"
                               "realism_per_target.json"),
        # per-target values for every bacterial method, not just the three
        # mRNA-GPT variants: Figure 5 shows the spread across the four targets
        "bacexp_realism_all": load("results/section5_method_comparison/raw/"
                                   "bacexp_realism_methods.json"),
        # No VAL+TEST-only evaluator exists for bacterial expression; this one
        # follows the standard protocol and so has seen the fine-tuning
        # sequences. It is biased toward the fine-tuned arm, and it still shows
        # no separation, which is the conservative direction for that result.
        "bacexp_oracle": load(f"{RUNS}/bacexp_sweep/generation_panel_n200/"
                              "bacexp_oracle_scores.json"),
        "bacexp_property_eval": load(f"{RUNS}/bacexp_sweep/generation_panel_n200/"
                                     "all_methods_property_eval.json")}
    # evaluator quality, for the caption
    out["evaluators"] = {
        "lgbm_stability": load("sft/lightgbm_stability/metrics.json"),
        "lgbm_expression": load("sft/lightgbm_expression/metrics.json"),
        "lgbm_bacexp": load("sft/lightgbm_bacexp/metrics.json"),
        "mrna_lm_stability": {"test_pearson": 0.391, "test_spearman": 0.389,
                              "protocol": "VAL+TEST only", "training_cai_max": 0.908,
                              "used_for_reported_numbers": True},
        "mrna_lm_stability_std": {"test_pearson": 0.4064, "test_spearman": 0.4070,
                                  "protocol": "TRAIN->train, VAL->select, TEST->report",
                                  "used_for_reported_numbers": False,
                                  "why_not": "trains on the split the fine-tuning set "
                                             "is drawn from, so it is not independent "
                                             "of the generator's training data"},
        "mrna_lm_expression": {"test_pearson": 0.620, "test_spearman": 0.636,
                               "protocol": "VAL+TEST only", "training_cai_max": 0.868,
                               "used_for_reported_numbers": True},
        "mrna_lm_bacexp_std": {"test_pearson": 0.295, "test_spearman": 0.311,
                               "protocol": "TRAIN->train, VAL->select, TEST->report",
                               "used_for_reported_numbers": True,
                               "caveat": "no VAL+TEST-only evaluator exists for this "
                                         "task; it is biased toward the fine-tuned arm "
                                         "and still shows no separation"}}

    # ---------- supplementary ----------
    # Bacterial learning-rate sweep (Fig S3): selection on VAL, reporting on TEST.
    out["bacexp_sweep"] = {
        split: load(f"{RUNS}/bacexp_sweep/sweep_realism_{split}.json")
        for split in ("val", "test")}
    # Nine yeast UTR contexts around one fixed CDS (Fig S4).
    out["utr_swap"] = load(f"{RUNS}/fungal_sft/generation_panel_n200/utr_swap_test.json")
    # 40 proteins, 65-816 aa (Fig S5).
    out["length_panel"] = load(f"{RUNS}/fungal_sft/generation_length/length_analysis.json")
    out["pretraining_curves"] = pretraining_curves()
    out["split_comparison"] = split_comparison()
    out["gc3"] = gc3_panel()

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")




def stability_all_methods_n200():
    """All-method stability realism, with the mRNA-GPT arms taken at n=200.

    The baseline designs were scored once, against the n=50 mRNA-GPT panel. The
    mRNA-GPT arms have since been regenerated at n=200 to match the other two
    tasks. Both files score against the same real high/low TEST genes -- the
    reference rows are identical to six decimals -- so the arms can be swapped in
    without rescoring the baselines, which are deterministic or fixed-n anyway.
    """
    base = load(f"{RUNS}/stability_sft/generation_panel/stability_realism_all_methods.json")
    n200 = load(f"{RUNS}/stability_sft/generation_panel_n200/realism_per_target_n200.json")
    if not base or not n200:
        return base
    for ref in ("REAL_high_test", "REAL_low_test"):
        a, b = base[ref]["_all"]["js_to_high"], n200[ref]["_all"]["js_to_high"]
        assert abs(a - b) < 1e-9, f"{ref}: reference differs, {a} vs {b}"
    for old_key, new_key in (("pretrained_archaea", "mrna_gpt_pretrained"),
                             ("stability_sft", "mrna_gpt_sft"),
                             ("stability_sft_LOW", "mrna_gpt_sft_LOW")):
        if new_key in n200:
            base[old_key] = n200[new_key]
            print(f"  stability all-methods: {old_key} <- n=200 panel")
    return base


def panel_identity():
    """Protein identity, CDS validity, diversity and memorisation, per panel cell.

    Computed here from the released FASTAs rather than read from an evaluator
    report, so that every column is something a reader can recompute. The
    nearest-neighbour identity that earlier versions reported is deliberately
    absent: it came from an MMseqs2 search that records "no hit" as 0% identity,
    which is not a similarity measurement. Exact matching against the arm's own
    fine-tuning set replaces it and is exact by construction.
    """
    import gzip
    import random
    from mrnagpt.vocab import GENETIC_CODE

    STOPS = {"UAA", "UAG", "UGA"}
    # Arm labels differ between panels because they were generated at different
    # times; the paths are listed explicitly rather than guessed. The fungal low
    # control lives in its own directory.
    ARMS = ("pretrained", "sft_high", "sft_low_control")
    FT_ARM = {"pretrained": None, "sft_high": "high", "sft_low_control": "low"}
    PANELS = {
        "stability": ("stability", f"{RUNS}/stability_sft/generation_panel_n200", {
            "pretrained": "mrna_gpt_pretrained.fasta",
            "sft_high": "mrna_gpt_sft.fasta",
            "sft_low_control": "low/mrna_gpt_sft_LOW.fasta"}),
        "expression": ("expression", f"{RUNS}/fungal_sft/generation_panel_n200", {
            "pretrained": "pretrained_eukaryote.fasta",
            "sft_high": "fungal_sft.fasta",
            "sft_low_control": f"{RUNS}/fungal_sft/generation_panel_low/"
                               "fungal_sft_low.fasta"}),
        "bacexp": ("bacteria_expression", f"{RUNS}/bacexp_sweep/generation_panel_n200", {
            "pretrained": "mrna_gpt_pretrained.fasta",
            "sft_high": "mrna_gpt_sft.fasta",
            "sft_low_control": "low/mrna_gpt_sft_LOW.fasta"}),
    }

    def norm(x):
        return x.strip().upper().replace(" ", "").replace("U", "T")

    def translate(codons):
        return "".join(GENETIC_CODE.get(c, "X") for c in codons
                       if GENETIC_CODE.get(c) not in (None, "*"))

    ft_cache: dict[str, set] = {}

    def finetune_set(tag, arm):
        key = f"{arm}_{tag}"
        if key not in ft_cache:
            path = f"sft/data/sft_{arm}_{tag}_train_codon.txt.gz"
            ft_cache[key] = ({norm(l) for l in gzip.open(path, "rt")}
                             if os.path.exists(path) else set())
        return ft_cache[key]

    out = {}
    for task, (tag, pdir, paths) in PANELS.items():
        man = load(f"{pdir}/generation_manifest.json")
        if not man:
            continue
        out[task] = {}
        for arm in ARMS:
            ft_arm = FT_ARM[arm]
            fname = paths[arm]
            path = fname if fname.startswith("/") else f"{pdir}/{fname}"
            if not os.path.exists(path):
                print(f"  missing panel fasta {path}")
                continue
            ft = finetune_set(tag, ft_arm) if ft_arm else set()
            groups: dict[str, list] = {}
            for sid, seq in read_fasta(path).items():
                groups.setdefault(sid.split("|")[1], []).append(seq)
            cells = {}
            for target, seqs in groups.items():
                prot = man["targets"][target]["protein"]
                cod = [to_codons(q) for q in seqs]
                ident = sum(translate(c) == prot for c in cod)
                valid = sum(1 for c in cod
                            if c and c[0] == "AUG" and c[-1] in STOPS
                            and not any(x in STOPS for x in c[:-1]))
                rng = random.Random(0)
                pairs = [(rng.randrange(len(cod)), rng.randrange(len(cod)))
                         for _ in range(1000)]
                sims = [sum(a == b for a, b in zip(cod[i], cod[j])) / len(cod[i])
                        for i, j in pairs if i != j]
                cells[target] = {
                    "n": len(seqs),
                    "protein_length_aa": man["targets"][target]["protein_len"],
                    "protein_identity_pct": 100.0 * ident / len(seqs),
                    "valid_cds_pct": 100.0 * valid / len(seqs),
                    "distinct_sequences_pct": 100.0 * len(set(seqs)) / len(seqs),
                    "mean_pairwise_codon_identity_pct":
                        100.0 * statistics.mean(sims) if sims else float("nan"),
                    "n_pairs_sampled": len(sims),
                    "exact_matches_to_finetuning_set":
                        sum(norm(q) in ft for q in seqs) if ft else 0,
                    "finetuning_set_size": len(ft)}
            out[task][arm] = cells
            tot = sum(c["n"] for c in cells.values())
            print(f"  identity {task}/{arm}: n={tot} "
                  f"distinct={sum(c['distinct_sequences_pct'] * c['n'] for c in cells.values()) / tot:.1f}% "
                  f"exact_ft_matches={sum(c['exact_matches_to_finetuning_set'] for c in cells.values())}")
    return out


def pretraining_curves():
    """Validation loss against step for the three pretrained models (Fig S1).

    These curves document which checkpoint each downstream analysis uses and
    where the reported step counts come from: they are the training log itself,
    so the step axis ends where training ended, and the best-validation point is
    marked.
    """
    out = {}
    for dom, per_epoch in (("archaea", 0.58e9), ("bacteria", 25.06e9),
                           ("eukaryote", 43.93e9)):
        path = f"{RUNS}/{dom}/metrics.csv"
        if not os.path.exists(path):
            print(f"  missing: {path}")
            continue
        step, vloss, tok = [], [], 0.0
        for r in csv.DictReader(open(path)):
            if r.get("val_loss"):
                step.append(int(r["global_step"]))
                vloss.append(float(r["val_loss"]))
            if r.get("tokens_seen"):
                tok = max(tok, float(r["tokens_seen"]))
        i = int(np.argmin(vloss))
        out[dom] = {"step": step, "val_loss": vloss,
                    "best_step": step[i], "best_val_loss": vloss[i],
                    "final_step": max(step), "tokens_seen": tok,
                    "epochs": tok / per_epoch}
        print(f"  curve {dom}: {len(step)} evals, best {vloss[i]:.4f} at step "
              f"{step[i]}, {tok/per_epoch:.2f} epochs")
    return out


def split_comparison():
    """Validation loss of the same architecture under the two splitting rules.

    The random-split run is configs/archaea_random.yaml, identical to
    configs/archaea.yaml except for which split file it reads. Its validation
    loss is far lower, but only because most of its validation set has a homolog
    in training -- which is the point of the panel.
    """
    out = {}
    for label, run in (("homology_aware", "archaea"), ("random", "archaea_random")):
        path = f"{RUNS}/{run}/metrics.csv"
        if not os.path.exists(path):
            print(f"  missing: {path}")
            continue
        # val_loss and train_probe_loss are written on different rows, so they
        # are collected as two series rather than one aligned list. The probe is
        # a held-out sample of the TRAINING distribution: it is what shows the
        # two runs are equally well fitted and that only the validation set
        # differs between them.
        steps, vloss, psteps, probe = [], [], [], []
        for r in csv.DictReader(open(path)):
            if r.get("val_loss"):
                steps.append(int(r["global_step"]))
                vloss.append(float(r["val_loss"]))
            if r.get("train_probe_loss"):
                psteps.append(int(r["global_step"]))
                probe.append(float(r["train_probe_loss"]))
        out[label] = {"step": steps, "val_loss": vloss,
                      "probe_step": psteps, "train_probe_loss": probe,
                      "best_val_loss": min(vloss) if vloss else None,
                      "final_train_probe_loss": probe[-1] if probe else None}
        print(f"  split {label}: {len(steps)} evals, best val loss "
              f"{min(vloss):.4f}" if vloss else f"  split {label}: no evals")
    return out


def gc3_panel():
    """GC3 of real validation sequences vs de novo generations, per domain.

    GC3 is the classic codon-usage signature and differs sharply between the
    three domains, so matching it is a direct read-out of whether each model
    learned its own domain's coding pattern rather than a generic codon prior.
    """
    import gzip
    import random
    out = {}
    for d in ("archaea", "bacteria", "eukaryote"):
        real = []
        with gzip.open(f"{DATA}/{d}/val_codon.txt.gz", "rt") as fh:
            for i, line in enumerate(fh):
                if i >= 20000:
                    break
                cod = line.split()
                if len(cod) >= 30:
                    third = [c[2] for c in cod if len(c) == 3]
                    real.append(sum(c in "GC" for c in third) / len(third))
        gen = []
        for s in read_fasta(f"{RUNS}/{d}/unconstrained.fasta").values():
            cod = to_codons(s)
            third = [c[2] for c in cod if len(c) == 3]
            if len(third) >= 30:
                gen.append(sum(c in "GC" for c in third) / len(third))
        random.Random(0).shuffle(real)
        out[d] = {"real": real[:4000], "generated": gen}
        print(f"  gc3 {d}: real n={len(real)} mean {sum(real)/len(real):.4f} | "
              f"gen n={len(gen)} mean {sum(gen)/max(len(gen),1):.4f}")
    return out


if __name__ == "__main__":
    main()
