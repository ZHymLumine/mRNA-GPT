#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=04:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_ld_stability
#PBS -o logs/lineardesign_stability.out
#PBS -e logs/lineardesign_stability.err
# LinearDesign baseline for the stability panel. Two codon-usage tables:
#   stability = built here from the top quartile of the mRNA-stability TRAIN
#               split, i.e. exactly the information the SFT model was given
#   human     = LinearDesign's own human table (the panel's actual host)
# lambda sweeps MFE-vs-CAI; the SFT model has no such knob, so sweeping it is
# the honest comparison.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=16
python -m sft.baseline_lineardesign \
    --train-csv sft/data/mrna_stability_train.csv \
    --tables ours,human --table-label stability \
    --lambdas 0,1,4 \
    --out-dir ${MRNA_GPT_RUNS}/stability_sft/generation_panel/baselines \
    --meta-name lineardesign_stability_meta.json
