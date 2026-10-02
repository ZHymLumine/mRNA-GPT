#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=04:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_lineardesign
#PBS -o logs/baseline_lineardesign.out
#PBS -e logs/baseline_lineardesign.err
# LinearDesign (Zhang et al., Nature 2023) on the target panel, lambda in {0,1,4},
# under two codon-usage tables: LinearDesign's shipped yeast table, and one built
# from our top-quartile fungal reference set. Single-threaded DP, ~n^3 in protein
# length -- the 676aa target dominates the runtime.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate mRNAdesigner3
cd "${MRNA_GPT_ROOT}"
python -m sft.baseline_lineardesign \
    --lambdas 0,1,4 --tables fungal,yeast \
    --out-dir ${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines
