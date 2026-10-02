#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=03:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_gen_stab200
#PBS -o logs/gen_stability_n200.out
#PBS -e logs/gen_stability_n200.err
# Regenerate the stability panel at n=200 per target, matching the expression
# panel, so every stochastic method is compared at the same sample size.
# Deterministic methods (CAI-max, LinearDesign at a fixed lambda, CodonBERT,
# native CDS) emit exactly one sequence by construction and cannot be aligned;
# they are handled by the percentile / best-of-n analysis instead.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
R=${MRNA_GPT_RUNS}
OUT=$R/stability_sft/generation_panel_n200
python -m sft.generate_target_panel --csv data/protein_sequences.csv \
    --pretrained-ckpt $R/archaea/model_best.pt --pretrained-label mrna_gpt_pretrained \
    --sft-ckpt $R/stability_sft/model_best.pt --sft-label mrna_gpt_sft \
    --n 200 --batch-size 25 --seed 42 --out-dir "$OUT"
python -m sft.generate_target_panel --csv data/protein_sequences.csv \
    --pretrained-ckpt $R/archaea/model_best.pt --pretrained-label mrna_gpt_pretrained \
    --sft-ckpt $R/stability_sft_low/model_best.pt --sft-label mrna_gpt_sft_LOW --only sft \
    --n 200 --batch-size 25 --seed 42 --out-dir "$OUT/low"
