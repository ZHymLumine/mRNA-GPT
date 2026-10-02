#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=03:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_mrnalm_bacexp
#PBS -o logs/mrna_lm_bacexp.out
#PBS -e logs/mrna_lm_bacexp.err
# Neural translation-efficiency evaluator, standard protocol (TRAIN -> train,
# VAL -> select, TEST -> report), matching the stability and expression ones.
#
# ONE CAVEAT TO CARRY INTO THE REPORT: mRNA-LM was pretrained on HUMAN
# transcripts, so applying it to E. coli is cross-domain. The fungal and
# stability evaluators do not have this problem. Its held-out test correlation
# is the honest measure of whether the transfer worked -- read it before
# quoting any ranking from it, and compare against the LightGBM TE predictor
# (test Pearson 0.453), which has no such mismatch.
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
python run_finetune_property.py --data data/bacteria_expression_lm.csv \
    -o ${MRNA_GPT_RUNS}/mrna_lm_bacexp_std
