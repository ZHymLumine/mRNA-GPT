#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=02:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_te_sft
#PBS -o logs/train_te_sft.out
#PBS -e logs/train_te_sft.err
# Fine-tunes mRNA-GPT-archaea on the high-translation-efficiency (TE > 2)
# subset. 439 SFT-train sequences / 15 steps per epoch, 1 GPU is plenty.
# Walltime is ~1.5x the 2.5 h max_hours budget: points are charged at job start
# on the REQUESTED walltime.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

SELF=${MRNA_GPT_ROOT}/pbs/train_te_sft.sh
source "${MRNA_GPT_ROOT}"/pbs/_train_common.sh

OUT=${MRNA_GPT_RUNS}/te_sft
mkdir -p "$OUT"

$TORCHRUN --standalone --nproc_per_node=1 -m mrnagpt.train \
    --config configs/te_sft.yaml

resubmit_if_needed "$SELF" "$OUT"
