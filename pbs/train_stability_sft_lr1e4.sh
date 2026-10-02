#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=03:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_stability_lr1e4
#PBS -o logs/train_stability_sft_lr1e4.out
#PBS -e logs/train_stability_sft_lr1e4.err
# Fine-tunes mRNA-GPT-archaea on the top-quartile (real measured mRNA stability)
# subset. 4,873 SFT-train sequences / 74 steps per epoch, 1 GPU is plenty.
# Walltime is ~1.5x the 2.5 h max_hours budget: points are charged at job start
# on the REQUESTED walltime.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

SELF=${MRNA_GPT_ROOT}/pbs/train_stability_sft_lr1e4.sh
source "${MRNA_GPT_ROOT}"/pbs/_train_common.sh

OUT=${MRNA_GPT_RUNS}/stability_sft_lr1e4
mkdir -p "$OUT"

$TORCHRUN --standalone --nproc_per_node=1 -m mrnagpt.train \
    --config configs/stability_sft_lr1e4.yaml

resubmit_if_needed "$SELF" "$OUT"
