#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=06:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_eval_bacexp
#PBS -o logs/eval_bacexp_methods.out
#PBS -e logs/eval_bacexp_methods.err
# Bacterial protein-expression panel: CAI, tAI, GC3, MFE, protein identity,
# CDS validity, novelty, diversity and the LightGBM expression predictor,
# for every method at the sample size each can actually produce.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=16
R=${MRNA_GPT_RUNS}
N=$R/bacexp_sft/generation_panel_n200
B=$N/baselines
P=$R/fungal_sft/generation_panel/baselines
python -m sft.evaluate_methods \
    --lgbm-dir sft/lightgbm_bacexp \
    --train-csv sft/data/bacteria_expression_train.csv \
    --property-name "protein expression" \
    --fasta mrna_gpt_pretrained=$N/mrna_gpt_pretrained.fasta \
    --fasta mrna_gpt_sft=$N/mrna_gpt_sft.fasta \
    --fasta mrna_gpt_sft_LOW=$N/low/mrna_gpt_sft_LOW.fasta \
    --fasta cai_sample=$B/cai_sample_n200.fasta \
    --fasta cai_max=$B/cai_max.fasta \
    --fasta lineardesign_l0=$B/lineardesign_bacexp_l0.fasta \
    --fasta lineardesign_l1=$B/lineardesign_bacexp_l1.fasta \
    --fasta lineardesign_l4=$B/lineardesign_bacexp_l4.fasta \
    --fasta native_cds=$P/native_cds_exact.fasta \
    --fasta codongpt=$P/codongpt.fasta \
    --fasta gemorna=$P/gemorna_n200.fasta \
    --fasta icodon=$P/icodon_n200.fasta \
    --fasta codonbert=$P/codonbert_fpp_fix_ids.fasta \
    --out $N/all_methods_property_eval.json \
    --md-out reports/bacexp_all_methods.md
