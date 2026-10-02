#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=01:30:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_eval_cmp
#PBS -o logs/eval_comparison.out
#PBS -e logs/eval_comparison.err
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate mRNAdesigner3
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=16
python -m sft.compare_pretrained_vs_sft
