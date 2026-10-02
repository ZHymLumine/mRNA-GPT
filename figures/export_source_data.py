"""Write one source-data table per figure panel into source_data/.

Every value that appears in a figure is read from figures/figure_data.json or
from the two descriptor tables beside it, which is the same path the plotting
scripts take, so a source-data file cannot disagree with the figure it belongs
to. Rebuild with:  python figures/export_source_data.py

Tables are long-format TSV: one row per plotted observation, with the grouping
columns spelled out rather than implied by position.
"""
from __future__ import annotations

import csv
import importlib.util as iu
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "source_data")
os.makedirs(OUT, exist_ok=True)

D = json.load(open(os.path.join(HERE, "figure_data.json")))
T4 = ["SARS_CoV_2", "Homo_sapiens", "Rabies_virus", "Zaire_ebolavirus"]
TASK_NAME = {"stability": "mRNA stability", "bacexp": "Bacterial expression",
             "expression": "Fungal expression"}
ARMS = ["pretrained", "sft_high", "sft_low_control"]
DENOVO_KEYS = {
    "stability": ("pretrained_archaea", "stability_sft", "stability_sft_LOW"),
    "bacexp": ("mrna_gpt_pretrained", "high_lr1e5", "low_lr1e5"),
    "expression": ("uncon_pretrained", "uncon_sft_HIGH", "uncon_sft_LOW"),
}
PANEL_ARM = {"pretrained": "mrna_gpt_pretrained", "sft_high": "mrna_gpt_sft",
             "sft_low_control": "mrna_gpt_sft_LOW"}
REAL_KEYS = {"stability": ("REAL_high_test", "REAL_low_test"),
             "expression": ("REAL_high_expression_test", "REAL_low_expression_test"),
             "bacexp": ("REAL_high_test", "REAL_low_test")}

_spec = iu.spec_from_file_location("f5", os.path.join(HERE, "fig5_methods.py"))
f5 = iu.module_from_spec(_spec)
_spec.loader.exec_module(f5)

INDEX: list[tuple[str, str, str]] = []


def write(name, panel, header, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else
                        (f"{v:.6g}" if isinstance(v, float) else v) for v in r])
    INDEX.append((panel, name, f"{len(rows)} rows"))
    print(f"  {name}  ({len(rows)} rows)")


# ----------------------------------------------------------------- Figure 2
def figure2():
    doms = ["archaea", "bacteria", "eukaryote"]
    write("figure2a_cross_domain_perplexity.tsv", "Figure 2a",
          ["pretrained_model", "validation_set", "perplexity_pad_excluded"],
          [[m, v, D["cross_domain_ppl"][m][v]] for m in doms for v in doms])

    rows = []
    for dom in doms:
        for i, th in enumerate(D["leakage"]["thresholds"]):
            for rule, key in (("random 90/10", "random"), ("homology-aware", "ours")):
                rows.append([dom, th.replace(">=", "≥"), rule,
                             D["leakage"][dom][key][i]])
    write("figure2b_split_leakage.tsv", "Figure 2b",
          ["domain", "nucleotide_identity_threshold", "split_rule",
           "validation_with_training_homolog_pct"], rows)

    rows = []
    for dom in doms:
        for src, key in (("real validation CDS", "real"),
                         ("de novo generated", "generated")):
            for v in D["gc3"][dom][key]:
                rows.append([dom, src, 100.0 * v])
    write("figure2c_gc3.tsv", "Figure 2c", ["domain", "sequence_set", "gc3_pct"], rows)

    write("figure2d_cds_validity.tsv", "Figure 2d",
          ["domain", "sequence_set", "syntactically_valid_cds_pct"],
          [[d, s, D["cds_validity"][d][k]] for d in doms
           for s, k in (("training corpus", "train"), ("de novo generated", "generated"))])

    rows = []
    for rule, key in (("homology-aware (ours)", "homology_aware"), ("random", "random")):
        b = D["split_comparison"][key]
        for s, v in zip(b["step"], b["val_loss"]):
            rows.append([rule, s, v])
    write("figure2e_split_validation_loss.tsv", "Figure 2e",
          ["split_rule", "global_step", "val_loss_nats_per_real_codon"], rows)


# ----------------------------------------------------------------- Figure 3
def figure3():
    rows = []
    for task in ("stability", "bacexp", "expression"):
        b = D["denovo"][task]
        gap = b["real_low"]["js_to_high"]
        lo, hi = b["real_low"]["llr_high_low"], b["real_high"]["llr_high_low"]
        for arm, key in zip(ARMS, DENOVO_KEYS[task]):
            v = b["arms"][key]
            rows.append([TASK_NAME[task], arm, v["n"], v["js_to_high"],
                         v["js_to_high"] / gap, v["llr_high_low"],
                         (v["llr_high_low"] - lo) / (hi - lo)])
        for lab, ref in (("REAL_high_quartile", "real_high"),
                         ("REAL_low_quartile", "real_low")):
            v = b[ref]
            rows.append([TASK_NAME[task], lab, v["n"], v["js_to_high"],
                         v["js_to_high"] / gap, v["llr_high_low"],
                         (v["llr_high_low"] - lo) / (hi - lo)])
    write("figure3ab_denovo_codon_profile.tsv", "Figure 3a, 3b",
          ["task", "arm", "n_sequences", "js_to_high_property_profile",
           "js_in_real_high_low_gap_units", "codon_log_likelihood_ratio",
           "llr_rescaled_0low_1high"], rows)

    summary, values = [], []
    for task in ("stability", "bacexp", "expression"):
        p = D["denovo_predictor"][task]
        lo, hi = p["real_low"], p["real_high"]
        for arm, key in zip(ARMS, ("pretrained", "sft", "sft_low")):
            if key not in p:
                continue
            summary.append([TASK_NAME[task], arm, len(p[key]["values"]),
                            p[key]["mean"], p[key]["sd"],
                            (p[key]["mean"] - lo) / (hi - lo)])
            for v in p[key]["values"]:
                values.append([TASK_NAME[task], arm, v, (v - lo) / (hi - lo)])
        summary.append([TASK_NAME[task], "REAL_high_quartile", None, hi, None, 1.0])
        summary.append([TASK_NAME[task], "REAL_low_quartile", None, lo, None, 0.0])
    write("figure3c_denovo_predictor_summary.tsv", "Figure 3c",
          ["task", "arm", "n_sequences", "predicted_mean", "predicted_sd",
           "predicted_rescaled_0low_1high"], summary)
    write("figure3d_denovo_predictor_values.tsv", "Figure 3d",
          ["task", "arm", "predicted", "predicted_rescaled_0low_1high"], values)

    src = list(csv.DictReader(open(os.path.join(HERE, "figure3_descriptors.tsv")),
                              delimiter="\t"))
    by = {(r["task"], r["arm"]): r for r in src}
    rows = []
    for task in ("stability", "bacexp", "expression"):
        hi, lo = by[(task, "[real_high genes]")], by[(task, "[real_low genes]")]
        for arm, label in zip(ARMS, ("Pretrained", "Fine-tuned (high)",
                                     "Fine-tuned (low, control)")):
            for m in ("cai", "tai", "gc3", "mfe_per_nt"):
                raw, h, l = float(by[(task, label)][m]), float(hi[m]), float(lo[m])
                rows.append([TASK_NAME[task], arm, m, raw, h, l, (raw - l) / (h - l)])
    write("figure3e_descriptors_rescaled.tsv", "Figure 3e",
          ["task", "arm", "descriptor", "value", "real_high_property_genes",
           "real_low_property_genes", "rescaled_0low_1high"], rows)


# ----------------------------------------------------------------- Figure 4
def figure4():
    rows = []
    for task in ("stability", "bacexp", "expression"):
        blk = D["panel"][task]
        dat = blk["data"]
        hi, lo = REAL_KEYS[task]
        gap = dat[lo]["_all"]["js_to_high"]
        llo, lhi = dat[lo]["_all"]["llr_high_low"], dat[hi]["_all"]["llr_high_low"]
        for arm in ARMS:
            m = PANEL_ARM[arm]
            for t in T4:
                v = dat[m][t]
                rows.append([TASK_NAME[task], arm, t, v["n"], v["js_to_high"],
                             v["js_to_high"] / gap, v["llr_high_low"],
                             (v["llr_high_low"] - llo) / (lhi - llo)])
        for lab, ref in (("REAL_high_quartile", hi), ("REAL_low_quartile", lo)):
            v = dat[ref]["_all"]
            rows.append([TASK_NAME[task], lab, "all_test_genes", v["n"],
                         v["js_to_high"], v["js_to_high"] / gap, v["llr_high_low"],
                         (v["llr_high_low"] - llo) / (lhi - llo)])
    write("figure4a-d_fixed_protein_codon_profile.tsv", "Figure 4a–4d",
          ["task", "arm", "target_protein", "n_variants",
           "js_to_high_property_profile", "js_in_real_high_low_gap_units",
           "codon_log_likelihood_ratio", "llr_rescaled_0low_1high"], rows)

    rows = []
    for task, arms in D["panel_identity"].items():
        for arm, cells in arms.items():
            for t in T4:
                c = cells.get(t)
                if not c:
                    continue
                rows.append([TASK_NAME[task], arm, t, c["protein_length_aa"], c["n"],
                             c["protein_identity_pct"], c["valid_cds_pct"],
                             c["distinct_sequences_pct"],
                             c["mean_pairwise_codon_identity_pct"],
                             c["n_pairs_sampled"],
                             c["exact_matches_to_finetuning_set"],
                             c["finetuning_set_size"] or None])
    write("figure4e_design_checks.tsv", "Figure 4e",
          ["task", "arm", "target_protein", "protein_length_aa", "n_variants",
           "protein_identity_pct", "valid_cds_pct", "distinct_sequences_pct",
           "mean_pairwise_codon_identity_pct", "n_pairs_sampled",
           "exact_matches_to_finetuning_set", "finetuning_set_size"], rows)


# ----------------------------------------------------------------- Figure 5
def figure5():
    AM = D["all_methods"]
    CTX = {"stability": "human_median", "expression": "adh1_yeast"}
    CAI_MAX = {"stability": 0.908, "expression": 0.868}
    rows = []
    for prop, realism, oracle in (("mRNA stability", AM["stability_realism"],
                                   AM["stability_oracle"]),
                                  ("Fungal expression", AM["expression_realism"],
                                   AM["expression_oracle"])):
        key = "stability" if prop.startswith("mRNA") else "expression"
        hi, lo = REAL_KEYS[key]
        tab = f5.realism_table(realism, hi, lo)
        orc = f5.oracle_table(oracle, CTX[key])
        gap = realism[lo]["_all"]["js_to_high"]
        for m in sorted(tab, key=lambda z: tab[z]["js"]):
            n = [realism[m][t]["n"] for t in realism[m] if t in T4]
            rows.append([prop, f5.PRETTY.get(m, m), m,
                         int(sum(n) / len(n)) if n else None,
                         tab[m]["js"] * gap, tab[m]["js"], tab[m]["cai"],
                         orc.get(m), int(tab[m]["cai"] > CAI_MAX[key])])
        for lab, ref in (("REAL_high_quartile", hi), ("REAL_low_quartile", lo)):
            v = realism[ref]["_all"]
            rows.append([prop, lab, ref, v["n"], v["js_to_high"],
                         v["js_to_high"] / gap,
                         v.get("cai_ref", v.get("cai_fungal")), None, 0])
    write("figure5a-d_method_comparison.tsv", "Figure 5a–5d",
          ["property", "method", "method_key", "n_per_target",
           "js_to_high_property_profile", "js_in_real_high_low_gap_units", "cai",
           "neural_evaluator_score", "outside_evaluator_training_cai_range"], rows)

    bp, bo = AM["bacexp_property_eval"], AM["bacexp_oracle"]
    rows = []
    for m in bp:
        for t in T4:
            c = bp[m].get(t)
            if not c:
                continue
            e = (bo["methods"].get(m) or {}).get(t, {}).get("ecoli_rbs", {})
            rows.append([f5.PRETTY.get(m, m), m, t, c["n"],
                         c.get("lgbm_predicted_expression", {}).get("mean"),
                         e.get("mean"), e.get("mannwhitney_p_vs_ref")])
    write("figure5ef_bacterial_evaluators.tsv", "Figure 5e, 5f",
          ["method", "method_key", "target_protein", "n_variants",
           "lgbm_predicted_expression", "neural_evaluator_score",
           "mannwhitney_p_vs_mrna_gpt_sft"], rows)


# ---------------------------------------------------------- Supplementary
def supplementary():
    rows = []
    for dom, b in D["pretraining_curves"].items():
        for s, v in zip(b["step"], b["val_loss"]):
            rows.append([dom, s, v, int(s == b["best_step"])])
    write("figureS1_pretraining_curves.tsv", "Supplementary Figure S1",
          ["domain", "global_step", "val_loss_nats_per_real_codon",
           "is_checkpoint_used"], rows)

    for src, name, panel in (("figure3_descriptors.tsv",
                              "figureS2_denovo_descriptors_raw.tsv",
                              "Supplementary Figure S2"),
                             ("figure5_descriptors.tsv",
                              "figureS6_panel_descriptors_raw.tsv",
                              "Supplementary Figure S6")):
        rows = list(csv.reader(open(os.path.join(HERE, src)), delimiter="\t"))
        write(name, panel, rows[0], rows[1:])

    rows = []
    for split in ("val", "test"):
        blk = D["bacexp_sweep"][split]
        gap = blk["REAL_low_test"]["_all"]["js_to_high"]
        for rate in ("1e5", "3e5", "1e4", "3e4"):
            for arm in ("high", "low"):
                v = blk[f"{arm}_lr{rate}"]["all"]
                rows.append([split.upper(), arm, rate.replace("e", "e-"), v["n"],
                             v["js_to_high"], v["js_to_high"] / gap,
                             v["llr_high_low"]])
        for lab in ("REAL_high_test", "REAL_low_test"):
            v = blk[lab]["_all"]
            rows.append([split.upper(), lab, "", v["n"], v["js_to_high"],
                         v["js_to_high"] / gap, v["llr_high_low"]])
    write("figureS3_bacexp_sweep.tsv", "Supplementary Figure S3",
          ["split", "arm", "peak_learning_rate", "n_sequences",
           "js_to_high_expression_profile", "js_in_real_high_low_gap_units",
           "codon_log_likelihood_ratio"], rows)

    u = D["utr_swap"]
    rows = []
    for key, d in u["sets"].items():
        for ctx, v in d["per_context_mean"].items():
            rows.append([key, ctx, u["contexts"][ctx]["group"],
                         u["contexts"][ctx].get("gene_id"), d["n_per_context"], v,
                         d["destabilizing_mean"], d["stable_mean"], d["delta"],
                         d["mannwhitney_p"]])
    write("figureS4_utr_swap.tsv", "Supplementary Figure S4",
          ["sequence_set", "utr_context", "context_group", "gene_id",
           "n_designs", "predicted_translation_rate_mean", "destabilising_mean",
           "stabilising_mean", "delta_destabilising_minus_stabilising",
           "mannwhitney_p"], rows)

    lp = D["length_panel"]
    rows = []
    for arm, cells in lp["arms"].items():
        for t, c in sorted(cells.items(), key=lambda kv: kv[1]["protein_length_aa"]):
            rows.append([arm, t, c["protein_length_aa"], c["n"], c["cai_mean"],
                         c["cai_sd"], c["js_to_high"], c["js_in_gap_units"],
                         c["llr_high_low"]])
    for r in lp["real_high_genes"]:
        rows.append(["real_high_expression_gene", "", r["protein_length_aa"], 1,
                     r["cai"], None, None, None, None])
    write("figureS5_length_panel.tsv", "Supplementary Figure S5",
          ["arm", "target_protein", "protein_length_aa", "n_variants", "cai_mean",
           "cai_sd", "js_to_high_expression_profile",
           "js_in_real_high_low_gap_units", "codon_log_likelihood_ratio"], rows)


def readme():
    lines = ["# Source data", "",
             "One table per figure panel. Each row is a plotted observation; the",
             "grouping columns are spelled out rather than implied by position.",
             "Distances marked `_in_real_high_low_gap_units` are divided by the",
             "Jensen–Shannon divergence between real high- and low-property genes of",
             "the same task, which is the unit used throughout the paper.", "",
             "Figure 1 is a schematic and has no source data.", "",
             "| panel | file | size |", "|---|---|---|"]
    order = {"Figure": 0, "Supplementary": 1}
    for panel, name, size in sorted(
            INDEX, key=lambda r: (order.get(r[0].split()[0], 2), r[1])):
        lines.append(f"| {panel} | `{name}` | {size} |")
    lines += ["", "Regenerate with `python figures/export_source_data.py`, which reads",
              "`figures/figure_data.json` — the same file the plotting scripts read."]
    open(os.path.join(OUT, "README.md"), "w").write("\n".join(lines) + "\n")
    print(f"  README.md  ({len(INDEX)} tables)")


if __name__ == "__main__":
    print("source_data/")
    figure2(); figure3(); figure4(); figure5(); supplementary(); readme()
    print("\ndone")
