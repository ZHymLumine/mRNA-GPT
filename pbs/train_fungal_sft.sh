#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=02:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_fungal_sft
#PBS -o logs/train_fungal_sft.out
#PBS -e logs/train_fungal_sft.err
# Fine-tunes mRNA-GPT-eukaryote on the top-quartile (real measured expression)
# fungal subset. Tiny dataset (1,149 SFT-train sequences), 1 GPU is plenty.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

SELF=${MRNA_GPT_ROOT}/pbs/train_fungal_sft.sh
source "${MRNA_GPT_ROOT}"/pbs/_train_common.sh

OUT=${MRNA_GPT_RUNS}/fungal_sft
mkdir -p "$OUT"

$TORCHRUN --standalone --nproc_per_node=1 -m mrnagpt.train \
    --config configs/fungal_sft.yaml

resubmit_if_needed "$SELF" "$OUT"
