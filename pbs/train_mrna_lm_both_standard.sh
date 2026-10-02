#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=05:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_mrnalm_std
#PBS -o logs/mrna_lm_standard.out
#PBS -e logs/mrna_lm_standard.err
# Neural property evaluators under the STANDARD supervised protocol: the
# dataset's own homology-clean TRAIN split for training (folds 1-3), VAL for
# model selection (fold 4), TEST for the reported held-out performance (fold 5).
#
# This replaces the earlier VAL+TEST-only evaluators. It uses the largest split,
# so the evaluator is stronger, and the reported test correlation is on the same
# TEST split every other evaluator is reported on. It is not circular: the SFT
# training subset is selected on MEASURED values with no predictor in the loop,
# so this predictor never influenced what the generative model was trained on.
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

python run_finetune_property.py --data data/mrna_stability_lm.csv \
    -o ${MRNA_GPT_RUNS}/mrna_lm_stability_std
python run_finetune_property.py --data data/fungal_expression_lm.csv \
    -o ${MRNA_GPT_RUNS}/mrna_lm_expression_std
