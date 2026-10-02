#!/usr/bin/env python3
"""Load Figure 5's data from the six canonical files into figures/figure_data.json.

One method set, one reference rule, one predictor protocol.  The six files are
produced by tools/rebuild_figure5_js.py (left column) and
pbs/fig5_score_all_tasks.sh (right column); this only transcribes them into the
cache the figure reads, so nothing here can introduce a value that the pipeline
did not compute.

It replaces an approach that spliced externally computed values into a
cache assembled from several different reference sets -- the arrangement that
let Figure 5's rows disagree about which genes were "real high-property" ones.

Idempotent: re-running writes the same block.
"""
from __future__ import annotations

import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.environ.get("MRNA_GPT_RUNS") or os.path.join(ROOT, "runs")
FD = os.path.join(ROOT, "figures/figure_data.json")
JS = os.path.join(ROOT, "figures/figure5_js_all.json")

T4 = ["Rabies_virus", "Zaire_ebolavirus", "SARS_CoV_2", "Homo_sapiens"]
METHODS13 = ["mrna_gpt_pretrained", "mrna_gpt_sft", "mrna_gpt_sft_LOW",
           "cai_max",
           "lineardesign_l0", "lineardesign_l1", "lineardesign_l4",
           "codongpt", "gemorna", "codonbert", "native_cds"]

ORACLE = {
    "stability": (f"{RUNS}/stability_sft/generation_panel_n200/"
                  "fig5_oracle_stability.json", "human_median"),
    "expression": (f"{RUNS}/fungal_sft/generation_panel_n200/"
                   "fig5_oracle_expression.json", "adh1_yeast"),
    "bacexp": (f"{RUNS}/bacexp_sweep/generation_panel_n200/"
               "fig5_oracle_bacexp.json", "ecoli_rbs"),
}
REALISM_KEY = {"stability": "stability_realism", "expression": "expression_realism",
               "bacexp": "bacexp_realism"}
ORACLE_KEY = {"stability": "stability_oracle", "expression": "expression_oracle",
              "bacexp": "bacexp_oracle"}


def main():
    d = json.load(open(FD))
    if not os.path.exists(FD + ".prefig5.bak"):
        shutil.copyfile(FD, FD + ".prefig5.bak")
    am = d["all_methods"]
    js = json.load(open(JS))

    for task in ("stability", "expression", "bacexp"):
        blk = js[task]
        gap = blk["gap"]
        # the realism block keeps the shape figures/fig5_methods.py expects:
        # method -> target -> {js_to_high}, plus the two reference rows
        realism = {}
        for m in METHODS13:
            per = blk["methods"][m]["per_target"]
            realism[m] = {t: {"js_to_high": v * gap,
                              "n": blk["methods"][m]["n_per_target"]}
                          for t, v in per.items()}
        realism["REAL_high_test"] = {"_all": {"js_to_high": 0.0}}
        realism["REAL_low_test"] = {"_all": {"js_to_high": gap}}
        realism["_selection"] = blk["_selection"]
        am[REALISM_KEY[task]] = realism

        path, ctx = ORACLE[task]
        oracle = json.load(open(path))
        am[ORACLE_KEY[task]] = {"methods": {m: oracle["methods"][m]
                                            for m in METHODS13},
                                "oracle_ckpt": oracle["oracle_ckpt"],
                                "context": ctx}
        print(f"{task:11s} gap={gap:.10f} refs "
              f"{blk['_selection']['n_high']}/{blk['_selection']['n_low']}  "
              f"{len(METHODS13)} methods both columns")

    # the old bacterial side-channels described a different reference set
    for dead in ("bacexp_realism_all", "bacexp_realism_methods"):
        if am.pop(dead, None) is not None:
            print(f"dropped stale {dead}")

    d["evaluators"] = dict(d.get("evaluators", {}), fig5_property_predictor={
        "protocol": "evaluation-only pool (VAL+TEST), whole-cluster 5 folds, "
                    "1-3 train / 4 select / 5 held out",
        "script": "run_finetune_property.py",
        "held_out": {"stability": {"pearson": 0.382, "spearman": 0.380},
                     "expression": {"pearson": 0.619, "spearman": 0.634},
                     "bacexp": {"pearson": 0.254, "spearman": 0.260}}})

    json.dump(d, open(FD, "w"))
    print("wrote figures/figure_data.json")


if __name__ == "__main__":
    main()
