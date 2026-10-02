#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=03:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_score_both
#PBS -o logs/score_both_oracles.out
#PBS -e logs/score_both_oracles.err
# Score every method -- now including codonGPT and CodonBERT (FPPGroup) -- under
# BOTH oracles, so each method can be read against its own host: the fungal
# evaluator for fungal-host methods, the human translation-rate evaluator for the
# human-host ones (GEMORNA, CodonBERT, codonGPT, LinearDesign+yeast table).
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
    --evaluator-note "The evaluator is a fungal-host expression evaluator: mRNA-LM LoRA fine-tuned on the VAL+TEST split of Fungal_expression.csv (whole-cluster 5-fold, fold 5 held out, test Pearson 0.620 / Spearman 0.636). It has seen neither the SFT training sequences nor their homologues, and has no CAI input features. Note that the CAI of its training data tops out at 0.868, so designs with CAI>0.87 are out-of-distribution extrapolation." \
    --fasta mrna_gpt_sft=${MRNA_GPT_RUNS}/fungal_sft/generation_panel_n200/fungal_sft.fasta --fasta mrna_gpt_sft_LOW=${MRNA_GPT_RUNS}/fungal_sft/generation_panel_low/fungal_sft_low.fasta --fasta mrna_gpt_pretrained=${MRNA_GPT_RUNS}/fungal_sft/generation_panel_n200/pretrained_eukaryote.fasta --fasta codongpt=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/codongpt.fasta --fasta codonbert_fpp_fix=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/codonbert_fpp_fix_norm.fasta --fasta gemorna=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/gemorna_n200.fasta --fasta cai_max=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/cai_max.fasta --fasta cai_sample=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/cai_sample.fasta --fasta ld_fungal_l2=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/lineardesign_fungal_l2.fasta --fasta ld_yeast_l4=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/lineardesign_yeast_l4.fasta \
    --out ${MRNA_GPT_RUNS}/fungal_sft/generation_panel_n200/fungal_oracle_all_methods.json \
    --md-out ${MRNA_GPT_ROOT}/reports/all_methods_fungal_oracle.md

python sft/score_panel_translation_rate.py \
    --ckpt ${MRNA_GPT_RUNS}/mrna_lm_tr/best_model.pt \
    --contexts adh1_yeast,human_median --reference mrna_gpt_sft \
    --fasta mrna_gpt_sft=${MRNA_GPT_RUNS}/fungal_sft/generation_panel_n200/fungal_sft.fasta --fasta mrna_gpt_sft_LOW=${MRNA_GPT_RUNS}/fungal_sft/generation_panel_low/fungal_sft_low.fasta --fasta mrna_gpt_pretrained=${MRNA_GPT_RUNS}/fungal_sft/generation_panel_n200/pretrained_eukaryote.fasta --fasta codongpt=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/codongpt.fasta --fasta codonbert_fpp_fix=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/codonbert_fpp_fix_norm.fasta --fasta gemorna=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/gemorna_n200.fasta --fasta cai_max=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/cai_max.fasta --fasta cai_sample=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/cai_sample.fasta --fasta ld_fungal_l2=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/lineardesign_fungal_l2.fasta --fasta ld_yeast_l4=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/lineardesign_yeast_l4.fasta \
    --out ${MRNA_GPT_RUNS}/fungal_sft/generation_panel_n200/human_oracle_all_methods.json \
    --md-out ${MRNA_GPT_ROOT}/reports/all_methods_human_oracle.md
