#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=12:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_archaea_40ep
#PBS -o logs/train_archaea_40ep.out
#PBS -e logs/train_archaea_40ep.err
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

SELF=${MRNA_GPT_ROOT}/pbs/train_archaea_40ep.sh
source "${MRNA_GPT_ROOT}"/pbs/_train_common.sh

OUT=${MRNA_GPT_RUNS}/archaea_40ep
mkdir -p "$OUT"
DATA=$(stage_data archaea)

$TORCHRUN --standalone --nproc_per_node=8 -m mrnagpt.train \
    --config configs/archaea_40ep.yaml --override data_dir="$DATA" out_dir="$OUT"

resubmit_if_needed "$SELF" "$OUT"
