#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=06:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_align_stab
#PBS -o logs/eval_aligned_stability.out
#PBS -e logs/eval_aligned_stability.err
# Stability panel with every STOCHASTIC method at n=200 per target.
# Deterministic methods (CAI-max, LinearDesign at a fixed lambda) emit one
# sequence by construction and cannot be aligned; they are kept in the table and
# read through the percentile / best-of-n analysis instead of a distributional
# test, which has no power at n=1.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=16
R=${MRNA_GPT_RUNS}
N=$R/stability_sft/generation_panel_n200
B=$R/stability_sft/generation_panel/baselines
python -m sft.evaluate_methods \
    --lgbm-dir sft/lightgbm_stability \
    --train-csv sft/data/mrna_stability_train.csv \
    --property-name stability \
    --fasta mrna_gpt_pretrained=$N/mrna_gpt_pretrained.fasta \
    --fasta mrna_gpt_sft=$N/mrna_gpt_sft.fasta \
    --fasta mrna_gpt_sft_LOW=$N/low/mrna_gpt_sft_LOW.fasta \
    --fasta cai_sample=$B/cai_sample_n200.fasta \
    --fasta cai_max=$B/cai_max.fasta \
    --fasta lineardesign_l0=$B/lineardesign_stability_l0.fasta \
    --fasta lineardesign_l1=$B/lineardesign_stability_l1.fasta \
    --fasta lineardesign_l4=$B/lineardesign_stability_l4.fasta \
    --fasta lineardesign_human_l0=$B/lineardesign_human_l0.fasta \
    --fasta lineardesign_human_l1=$B/lineardesign_human_l1.fasta \
    --fasta lineardesign_human_l4=$B/lineardesign_human_l4.fasta \
    --out $N/all_methods_property_eval_aligned.json \
    --md-out reports/stability_all_methods_aligned.md
