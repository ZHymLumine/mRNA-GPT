#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=04:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_archaea
#PBS -o logs/train_archaea.out
#PBS -e logs/train_archaea.err
# Walltime is sized at ~1.5x the measured cost, not padded to the 168 h the queue
# allows: points are charged at job start on the *requested* walltime, so asking
# for 24 h to run for 2 h wastes 22 h of budget.  Resume is unconditional, so this
# file can be qsub'ed verbatim.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

SELF=${MRNA_GPT_ROOT}/pbs/train_archaea.sh
source "${MRNA_GPT_ROOT}"/pbs/_train_common.sh

OUT=${MRNA_GPT_RUNS}/archaea
mkdir -p "$OUT"
DATA=$(stage_data archaea)

$TORCHRUN --standalone --nproc_per_node=8 -m mrnagpt.train \
    --config configs/archaea.yaml --override data_dir="$DATA" out_dir="$OUT"

resubmit_if_needed "$SELF" "$OUT"
