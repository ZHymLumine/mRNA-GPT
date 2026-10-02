#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=02:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_icodon_sup
#PBS -o logs/eval_icodon_supplement.out
#PBS -e logs/eval_icodon_supplement.err
# iCodon alone at n=200, to be merged into the expression all-methods table.
# It is split out because its 200-seed run finished at the same moment the main
# expression evaluation started, so that job's method list did not include it.
# mrna_gpt_sft is re-scored alongside it purely as a consistency check: its
# numbers must match the main run, since it is the identical FASTA.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate mRNAdesigner3
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=16
R=${MRNA_GPT_RUNS}
python -m sft.evaluate_methods \
    --lgbm-dir sft/lightgbm_expression \
    --train-csv sft/data/fungal_expression_train.csv \
    --property-name expression \
    --fasta icodon=$R/fungal_sft/generation_panel/baselines/icodon_n200.fasta \
    --fasta mrna_gpt_sft=$R/fungal_sft/generation_panel_n200/fungal_sft.fasta \
    --out $R/fungal_sft/generation_panel_n200/icodon_supplement.json \
    --md-out reports/expression_icodon_supplement.md
