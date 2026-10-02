#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=02:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_score_fungal
#PBS -o logs/score_panel_fungal.out
#PBS -e logs/score_panel_fungal.err
# Re-score every method with the host-matched fungal expression evaluator
# (mRNA-LM fine-tuned on VAL+TEST only, held-out fold Pearson 0.620 / Spearman
# 0.636, no CAI input feature). Replaces the LightGBM column, which correlates
# with CAI at r=0.88 and therefore cannot rank CAI-maximizing methods.
# Only the ADH1 context is scored: this evaluator was trained with that context
# held constant, so any other UTR pair is off-distribution for it.
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
python sft/score_panel_translation_rate.py \
    --ckpt ${MRNA_GPT_RUNS}/mrna_lm_fungal/best_model.pt \
    --contexts adh1_yeast --reference mrna_gpt_sft \
    --fasta mrna_gpt_pretrained=${MRNA_GPT_RUNS}/fungal_sft/generation_panel_n200/pretrained_eukaryote.fasta \
    --fasta mrna_gpt_sft=${MRNA_GPT_RUNS}/fungal_sft/generation_panel_n200/fungal_sft.fasta \
    --fasta mrna_gpt_sft_LOW=${MRNA_GPT_RUNS}/fungal_sft/generation_panel_low/fungal_sft_low.fasta \
    --fasta gemorna=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/gemorna_n200.fasta \
    --fasta cai_max=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/cai_max.fasta \
    --fasta cai_sample=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/cai_sample.fasta \
    --fasta ld_fungal_l0=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/lineardesign_fungal_l0.fasta \
    --fasta ld_fungal_l1p5=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/lineardesign_fungal_l1.5.fasta \
    --fasta ld_fungal_l2=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/lineardesign_fungal_l2.fasta \
    --fasta ld_fungal_l4=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/lineardesign_fungal_l4.fasta \
    --fasta ld_fungal_l10=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/lineardesign_fungal_l10.fasta \
    --out ${MRNA_GPT_RUNS}/fungal_sft/generation_panel_n200/fungal_oracle_scores.json \
    --md-out ${MRNA_GPT_ROOT}/reports/panel_fungal_oracle.md
