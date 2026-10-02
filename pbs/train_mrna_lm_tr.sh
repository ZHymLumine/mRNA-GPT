#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=03:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_mrnalm_tr
#PBS -o logs/mrna_lm_tr.out
#PBS -e logs/mrna_lm_tr.err
# One-time LoRA fine-tune of the published mRNA-LM model on its own bundled
# translation-rate task (the oracle used by Li et al. mRNA-GPT), used unmodified afterward as a fixed
# downstream evaluator (never fine-tuned on our fungal data).
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

python run_finetune_tr.py -o ${MRNA_GPT_RUNS}/mrna_lm_tr
