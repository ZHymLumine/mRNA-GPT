#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=12:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_bacteria
#PBS -o logs/train_bacteria.out
#PBS -e logs/train_bacteria.err
# Walltime is sized at ~1.5x the measured cost, not padded to the 168 h the queue
# allows: points are charged at job start on the *requested* walltime, so asking
# for 24 h to run for 2 h wastes 22 h of budget.  Resume is unconditional, so this
# file can be qsub'ed verbatim.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

SELF=${MRNA_GPT_ROOT}/pbs/train_bacteria.sh
source "${MRNA_GPT_ROOT}"/pbs/_train_common.sh

OUT=${MRNA_GPT_RUNS}/bacteria
mkdir -p "$OUT"
DATA=$(stage_data bacteria)

$TORCHRUN --standalone --nproc_per_node=8 -m mrnagpt.train \
    --config configs/bacteria.yaml --override data_dir="$DATA" out_dir="$OUT"

resubmit_if_needed "$SELF" "$OUT"
