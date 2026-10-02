#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=04:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_bacexp_sweep
#PBS -o logs/train_bacexp_sweep.out
#PBS -e logs/train_bacexp_sweep.err
# Learning-rate sweep for bacterial-expression SFT: 4 peak LRs x 2 arms.
# Rationale and what is deliberately NOT swept: configs/sweep/make_bacexp_sweep.py
#
# 8 runs x ~10 min (the original 10-epoch run took 8.5 min for 297 steps) is
# about 80 min; 4 h requested because points are charged on the REQUESTED
# walltime at job start and a single GPU node is cheap at this size.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

set -euo pipefail
SELF=${MRNA_GPT_ROOT}/pbs/train_bacexp_sweep.sh
source "${MRNA_GPT_ROOT}"/pbs/_train_common.sh
cd "${MRNA_GPT_ROOT}"

for lr in 1e5 3e5 1e4 3e4; do
  for arm in high low; do
    OUT=${MRNA_GPT_RUNS}/bacexp_sweep/${arm}_lr${lr}
    if [ -f "$OUT/DONE" ]; then echo "skip ${arm}_lr${lr} (DONE)"; continue; fi
    mkdir -p "$OUT"
    echo "=== training ${arm}_lr${lr} ==="
    $TORCHRUN --standalone --nproc_per_node=1 -m mrnagpt.train \
        --config configs/sweep/bacexp_${arm}_lr${lr}.yaml
  done
done
echo "SWEEP TRAINING COMPLETE"
