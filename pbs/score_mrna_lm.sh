#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=00:45:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_mrnalm_score
#PBS -o logs/mrna_lm_score.out
#PBS -e logs/mrna_lm_score.err
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
export CUDA_VISIBLE_DEVICES=0

OUT=${MRNA_GPT_RUNS}/fungal_sft/generation
CKPT=${MRNA_GPT_RUNS}/mrna_lm_5class/best_model.pt

python -m evaluate.mrna_lm_score --checkpoint "$CKPT" \
    --fasta "$OUT/pretrained_eukaryote.fasta" --out "$OUT/mrna_lm_score_pretrained_eukaryote.jsonl"

python -m evaluate.mrna_lm_score --checkpoint "$CKPT" \
    --fasta "$OUT/fungal_sft.fasta" --out "$OUT/mrna_lm_score_fungal_sft.jsonl"
