#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=04:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_ld_bacexp
#PBS -o logs/lineardesign_bacexp.out
#PBS -e logs/lineardesign_bacexp.err
# LinearDesign for the bacterial protein-expression panel. Only the "ours" table
# is used: it is built from the same E >= 4 subset the model was fine-tuned on,
# so LinearDesign receives the identical codon-preference information. The tool
# ships human and yeast tables only, neither of which is a bacterial host.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=16
python -m sft.baseline_lineardesign \
    --train-csv sft/data/bacteria_expression_train.csv --threshold 4 \
    --tables ours --table-label bacexp --lambdas 0,1,4 \
    --out-dir ${MRNA_GPT_RUNS}/bacexp_sft/generation_panel_n200/baselines \
    --meta-name lineardesign_bacexp_meta.json
