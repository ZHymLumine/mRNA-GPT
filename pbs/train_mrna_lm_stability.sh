#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=04:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_mrnalm_stab
#PBS -o logs/mrna_lm_stability.out
#PBS -e logs/mrna_lm_stability.err
# NEURAL stability evaluator: one-time LoRA fine-tune of mRNA-LM on the mRNA
# stability task, host-matched to the panel (human UTR context), trained on the
# VAL+TEST splits ONLY. Used unmodified afterwards as a fixed downstream oracle.
#
# 8,296 sequences / 5,622 whole-cluster folds -- 4x the fungal evaluator's
# training set, so 4 h walltime rather than 3.
#
# It exists because the LightGBM stability predictor takes CAI as an input
# feature and therefore cannot fairly rank CAI-shifting methods. This one reads
# the codon sequence.
#
# It is NOT mRNA-LM's bundled half-life head: that head's training data shares
# 76.9% of its CDSs exactly with mRNA_Stability.csv, which would put 30.6% of
# our SFT training subset inside the evaluator's supervised training set.
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

cd ${MRNA_GPT_EXTERNAL}/mRNA-LM
export MRNA_LM_WEIGHTS=${MRNA_GPT_EXTERNAL}/mRNA-LM/weights/final
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0

python run_finetune_property.py \
    --data data/mrna_stability_lm.csv \
    -o ${MRNA_GPT_RUNS}/mrna_lm_stability
