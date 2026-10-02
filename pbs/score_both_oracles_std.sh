#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=03:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_score_std
#PBS -o logs/score_both_oracles_std.out
#PBS -e logs/score_both_oracles_std.err
# Re-score both property panels with the standard-protocol neural evaluators
# (TRAIN -> train, VAL -> select, TEST -> report; held-out test Pearson 0.406 for
# stability and 0.648 for expression). The FASTA list matches
# pbs/run_all_methods_property_eval.sh exactly, so the neural-evaluator column
# and the CAI/tAI/GC3/MFE/predictor columns describe the same sequences.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"
: "${MRNA_GPT_EXTERNAL:=$MRNA_GPT_ROOT/external}"

set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.1/12.1.1 2>/dev/null || module load cuda/12.6/12.6.1
source ~/.bashrc
# Relies on the conda environment that is already active (it must carry the
# mRNA-LM dependencies); set MRNA_LM_ENV to activate a named one instead.
if [ -n "${MRNA_LM_ENV:-}" ]; then conda activate "$MRNA_LM_ENV"; fi
cd "${MRNA_GPT_ROOT}"
export MRNA_LM_WEIGHTS=${MRNA_GPT_EXTERNAL}/mRNA-LM/weights/final
export MRNA_LM_REPO=${MRNA_GPT_EXTERNAL}/mRNA-LM
export TOKENIZERS_PARALLELISM=true
R=${MRNA_GPT_RUNS}

NOTE_COMMON="The evaluator is fine-tuned under the standard supervised protocol on this dataset's own homology-clean split: train on the TRAIN split (folds 1-3), select on VAL (fold 4), report on TEST (fold 5). It has no CAI input features. The SFT training subset was selected on **measured values**, with no predictor involved in the selection, so this predictor never influenced what the generative model was trained on. "

# ---------------- stability ----------------
P=$R/stability_sft/generation_panel; B=$P/baselines
python sft/score_panel_translation_rate.py \
    --ckpt $R/mrna_lm_stability_std/best_model.pt \
    --contexts human_median --reference mrna_gpt_sft \
    --evaluator-note "mRNA stability evaluator (held-out TEST Pearson 0.406 / Spearman 0.407). ${NOTE_COMMON}All records share the fixed human UTR context ENST00000320934." \
    --fasta mrna_gpt_pretrained=$P/pretrained_archaea.fasta \
    --fasta mrna_gpt_sft=$P/stability_sft.fasta \
    --fasta mrna_gpt_sft_LOW=$P/low/stability_sft_low.fasta \
    --fasta cai_max=$B/cai_max.fasta \
    --fasta cai_sample=$B/cai_sample.fasta \
    --fasta lineardesign_l0=$B/lineardesign_stability_l0.fasta \
    --fasta lineardesign_l1=$B/lineardesign_stability_l1.fasta \
    --fasta lineardesign_l4=$B/lineardesign_stability_l4.fasta \
    --fasta lineardesign_human_l0=$B/lineardesign_human_l0.fasta \
    --fasta lineardesign_human_l1=$B/lineardesign_human_l1.fasta \
    --fasta lineardesign_human_l4=$B/lineardesign_human_l4.fasta \
    --out $P/stability_oracle_scores_std.json \
    --md-out reports/stability_panel_neural_oracle_std.md

# ---------------- expression ----------------
P=$R/fungal_sft/generation_panel; B=$P/baselines
python sft/score_panel_translation_rate.py \
    --ckpt $R/mrna_lm_expression_std/best_model.pt \
    --contexts adh1_yeast --reference mrna_gpt_sft \
    --evaluator-note "Fungal expression evaluator (held-out TEST Pearson 0.648 / Spearman 0.650). ${NOTE_COMMON}All records share the fixed yeast ADH1 UTR context." \
    --fasta mrna_gpt_pretrained=$P/pretrained_eukaryote.fasta \
    --fasta mrna_gpt_sft=$P/fungal_sft.fasta \
    --fasta mrna_gpt_sft_LOW=$R/fungal_sft/generation_panel_low/fungal_sft_low.fasta \
    --fasta cai_max=$P/cai_max.fasta \
    --fasta cai_sample=$P/cai_sample.fasta \
    --fasta lineardesign_l0=$B/lineardesign_fungal_l0.fasta \
    --fasta lineardesign_l1=$B/lineardesign_fungal_l1.fasta \
    --fasta lineardesign_l4=$B/lineardesign_fungal_l4.fasta \
    --fasta lineardesign_yeast_l0=$B/lineardesign_yeast_l0.fasta \
    --fasta lineardesign_yeast_l1=$B/lineardesign_yeast_l1.fasta \
    --fasta lineardesign_yeast_l4=$B/lineardesign_yeast_l4.fasta \
    --fasta gemorna=$B/gemorna.fasta \
    --fasta codongpt=$B/codongpt.fasta \
    --fasta icodon=$B/icodon.fasta \
    --fasta codonbert=$B/codonbert_fpp_fix_ids.fasta \
    --fasta native_cds=$B/native_cds_exact.fasta \
    --out $P/expression_oracle_scores_std.json \
    --md-out reports/expression_panel_neural_oracle_std.md
