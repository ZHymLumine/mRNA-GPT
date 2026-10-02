#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=06:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_gemorna200
#PBS -o logs/baseline_gemorna200.out
#PBS -e logs/baseline_gemorna200.err
# GEMORNA zero-shot CDS generation on the target panel, 200 variants per target (the setting used by Li et al.).
# Runs in the dedicated gemorna env (cpython-310 .so + torchtext 0.6 vocab
# pickles). CPU-only and ~0.4 s/residue, so the 200 generations are spread over
# a 16-process pool; the 676aa target dominates.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=1
# Expects the GEMORNA conda environment to be active.
python sft/baseline_gemorna.py \
    --n 200 --workers 16 \
    --out ${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/gemorna_n200.fasta
