#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=04:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_fig5_score
#PBS -o logs/fig5_score_all_tasks.out
#PBS -e logs/fig5_score_all_tasks.err
# Figure 5's right column: the same thirteen methods, scored on each task by
# that task's property predictor.  All three predictors were trained by one
# script under one protocol (pbs/train_mrna_lm_valtest_all.sh), so the column
# is the same measurement in every row; only the checkpoint and the fixed UTR
# context change, and the context must be the one its predictor was trained
# under or the 5'/3' encoders see off-distribution input.
#
# Held out: stability 0.382, fungal 0.619, bacteria 0.254 (Pearson).
#
# The five host-fixed tools carry their codon preference in released weights or
# in the source organism, so one design set serves all three tasks; the same
# FASTA therefore appears in all three blocks.
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
FB=$R/fungal_sft/generation_panel/baselines
HF=(--fasta codongpt=$FB/codongpt.fasta
    --fasta gemorna=$FB/gemorna_n200.fasta
    --fasta icodon=$FB/icodon_n200.fasta
    --fasta codonbert=$FB/codonbert_fpp_fix_norm.fasta
    --fasta native_cds=$FB/native_cds.fasta)

# ---------------- mRNA stability ----------------
P=$R/stability_sft; B=$P/generation_panel/baselines; N=$P/generation_panel_n200
python sft/score_panel_translation_rate.py \
    --ckpt $R/mrna_lm_stability_vt/best_model.pt \
    --contexts human_median --reference mrna_gpt_sft \
    --evaluator-note "Stability property predictor: mRNA-LM with LoRA, trained on an evaluation-only pool (VAL+TEST, whole-cluster 5-fold; folds 1-3 train / fold 4 model selection / fold 5 held out), held-out Pearson 0.382 / Spearman 0.380. Fixed human UTR context." \
    --fasta mrna_gpt_pretrained=$N/mrna_gpt_pretrained.fasta \
    --fasta mrna_gpt_sft=$N/mrna_gpt_sft.fasta \
    --fasta mrna_gpt_sft_LOW=$N/low/mrna_gpt_sft_LOW.fasta \
    --fasta cai_max=$B/cai_max.fasta \
    --fasta cai_sample=$B/cai_sample_n200.fasta \
    --fasta lineardesign_l0=$B/lineardesign_stability_l0.fasta \
    --fasta lineardesign_l1=$B/lineardesign_stability_l1.fasta \
    --fasta lineardesign_l4=$B/lineardesign_stability_l4.fasta \
    "${HF[@]}" \
    --out $N/fig5_oracle_stability.json \
    --md-out reports/fig5_oracle_stability.md

# ---------------- fungal transcript expression ----------------
P=$R/fungal_sft; N=$P/generation_panel_n200
python sft/score_panel_translation_rate.py \
    --ckpt $R/mrna_lm_fungal_vt/best_model.pt \
    --contexts adh1_yeast --reference mrna_gpt_sft \
    --evaluator-note "Fungal transcript expression property predictor: same script, same protocol, held-out Pearson 0.619 / Spearman 0.634. Fixed yeast ADH1 UTR context." \
    --fasta mrna_gpt_pretrained=$N/pretrained_eukaryote.fasta \
    --fasta mrna_gpt_sft=$N/fungal_sft.fasta \
    --fasta mrna_gpt_sft_LOW=$P/generation_panel_low/fungal_sft_low.fasta \
    --fasta cai_max=$P/generation_panel/cai_max.fasta \
    --fasta cai_sample=$FB/cai_sample_n200.fasta \
    --fasta lineardesign_l0=$FB/lineardesign_fungal_l0.fasta \
    --fasta lineardesign_l1=$FB/lineardesign_fungal_l1.fasta \
    --fasta lineardesign_l4=$FB/lineardesign_fungal_l4.fasta \
    "${HF[@]}" \
    --out $N/fig5_oracle_expression.json \
    --md-out reports/fig5_oracle_expression.md

# ---------------- bacteria protein expression ----------------
N=$R/bacexp_sweep/generation_panel_n200; B=$R/bacexp_sft/generation_panel_n200/baselines
python sft/score_panel_translation_rate.py \
    --ckpt $R/mrna_lm_bacexp_vt/best_model.pt \
    --contexts ecoli_rbs --reference mrna_gpt_sft \
    --evaluator-note "Bacterial protein expression property predictor: same script, same protocol, held-out Pearson 0.254 / Spearman 0.260 -- the weakest of the three, its training pool holding only 1,139 records. Fixed E. coli SD / terminator context." \
    --fasta mrna_gpt_pretrained=$N/mrna_gpt_pretrained.fasta \
    --fasta mrna_gpt_sft=$N/mrna_gpt_sft.fasta \
    --fasta mrna_gpt_sft_LOW=$N/low/mrna_gpt_sft_LOW.fasta \
    --fasta cai_max=$B/cai_max.fasta \
    --fasta cai_sample=$B/cai_sample_n200.fasta \
    --fasta lineardesign_l0=$B/lineardesign_bacexp_l0.fasta \
    --fasta lineardesign_l1=$B/lineardesign_bacexp_l1.fasta \
    --fasta lineardesign_l4=$B/lineardesign_bacexp_l4.fasta \
    "${HF[@]}" \
    --out $N/fig5_oracle_bacexp.json \
    --md-out reports/fig5_oracle_bacexp.md

echo "DONE fig5_score_all_tasks"
