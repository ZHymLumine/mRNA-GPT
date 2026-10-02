#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=04:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_ld_te
#PBS -o logs/lineardesign_te.out
#PBS -e logs/lineardesign_te.err
# LinearDesign for the TE panel. Only the "ours" table is used: it is built from
# the same TE > 2 subset the model was fine-tuned on, so LinearDesign gets the
# identical codon-preference information. LinearDesign ships human and yeast
# tables only -- neither is an E. coli host, so neither is included.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate mRNAdesigner3
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=16
python -m sft.baseline_lineardesign \
    --train-csv sft/data/ecoli_te_train.csv --threshold 2 \
    --tables ours --table-label te --lambdas 0,1,4 \
    --out-dir ${MRNA_GPT_RUNS}/te_sft/generation_panel_n200/baselines \
    --meta-name lineardesign_te_meta.json
