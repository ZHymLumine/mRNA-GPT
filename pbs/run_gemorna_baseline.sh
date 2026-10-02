#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=03:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_gemorna
#PBS -o logs/baseline_gemorna.out
#PBS -e logs/baseline_gemorna.err
# GEMORNA zero-shot CDS generation on the target panel, 50 variants per target.
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
    --n 50 --workers 16 \
    --out ${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/gemorna.fasta
