#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=12:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_icodon200
#PBS -o logs/icodon_n200.out
#PBS -e logs/icodon_n200.err
# iCodon at 200 random seeds per target, to match the sample size of the other
# stochastic methods. Its genetic algorithm is seed-dependent, so extra seeds
# give a genuine sample rather than repeats of one answer. 200 seeds x 4 targets
# x 15 GA iterations is the reason for the long walltime.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
cd "${MRNA_GPT_ROOT}"
B=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines
# Uses Rscript from the active conda environment (iCodon R package required).
Rscript sft/baseline_icodon.R \
    $B/icodon_start.fasta $B/icodon_n200.fasta 200 human 15
