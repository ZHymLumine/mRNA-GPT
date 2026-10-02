#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=01:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_eval_cmp_constrained
#PBS -o logs/eval_comparison_constrained.out
#PBS -e logs/eval_comparison_constrained.err
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate mRNAdesigner3
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=16
python -m sft.compare_pretrained_vs_sft \
    --pretrained-fasta ${MRNA_GPT_RUNS}/fungal_sft/generation_constrained/pretrained_eukaryote_constrained.fasta \
    --sft-fasta ${MRNA_GPT_RUNS}/fungal_sft/generation_constrained/fungal_sft_constrained.fasta \
    --out ${MRNA_GPT_RUNS}/fungal_sft/generation_constrained/evaluator_comparison.json
