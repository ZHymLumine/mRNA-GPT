#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=04:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_eval_allmethods
#PBS -o logs/eval_all_methods.out
#PBS -e logs/eval_all_methods.err
# One evaluator suite over every method on the target panel: mRNA-GPT before and
# after fungal SFT, classical CAI optimization (deterministic max and weighted
# sampling), GEMORNA, and LinearDesign at three lambdas under two codon-usage
# tables. CAI/tAI/LightGBM are recomputed here with OUR fungal reference for
# every method, so the numbers are comparable across methods -- LinearDesign's
# self-reported CAI uses whichever table it was given and is not.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=16
python -m sft.evaluate_methods \
    --fasta mrna_gpt_pretrained=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/pretrained_eukaryote.fasta \
    --fasta mrna_gpt_sft=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/fungal_sft.fasta \
    --fasta cai_max=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/cai_max.fasta \
    --fasta cai_sample=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/cai_sample.fasta \
    --fasta gemorna=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/gemorna.fasta \
    --fasta ld_fungal_l0=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/lineardesign_fungal_l0.fasta \
    --fasta ld_fungal_l1=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/lineardesign_fungal_l1.fasta \
    --fasta ld_fungal_l4=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/lineardesign_fungal_l4.fasta \
    --fasta ld_yeast_l0=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/lineardesign_yeast_l0.fasta \
    --fasta ld_yeast_l1=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/lineardesign_yeast_l1.fasta \
    --fasta ld_yeast_l4=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/lineardesign_yeast_l4.fasta \
    --out ${MRNA_GPT_RUNS}/fungal_sft/generation_panel/all_methods_per_target.json \
    --md-out ${MRNA_GPT_ROOT}/reports/target_panel_all_methods.md
