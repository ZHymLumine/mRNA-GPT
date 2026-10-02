#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=01:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_gen_panel200
#PBS -o logs/gen_target_panel200.out
#PBS -e logs/gen_target_panel200.err
# Protein-constrained generation for the four real target proteins in
# data/protein_sequences.csv, 200 independent synonymous variants each (matching the baselines n), from both
# the pretrained eukaryote checkpoint and the fungal-SFT checkpoint. Paired by
# construction: same amino-acid sequence on both sides, so only codon choice
# can move any downstream metric.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"

cd "${MRNA_GPT_ROOT}"
python -m sft.generate_target_panel \
    --csv data/protein_sequences.csv \
    --n 200 --batch-size 25 --seed 42 \
    --out-dir ${MRNA_GPT_RUNS}/fungal_sft/generation_panel_n200
