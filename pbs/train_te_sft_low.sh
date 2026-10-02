#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=02:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_te_sft_low
#PBS -o logs/train_te_sft_low.out
#PBS -e logs/train_te_sft_low.err
# NEGATIVE CONTROL for pbs/train_te_sft.sh: same procedure and
# hyperparameters, bottom quartile of measured stability instead of the top.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

SELF=${MRNA_GPT_ROOT}/pbs/train_te_sft_low.sh
source "${MRNA_GPT_ROOT}"/pbs/_train_common.sh

OUT=${MRNA_GPT_RUNS}/te_sft_low
mkdir -p "$OUT"

$TORCHRUN --standalone --nproc_per_node=1 -m mrnagpt.train \
    --config configs/te_sft_low.yaml

resubmit_if_needed "$SELF" "$OUT"
