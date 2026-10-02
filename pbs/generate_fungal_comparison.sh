#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=01:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_gen_fungal_cmp
#PBS -o logs/gen_fungal_comparison.out
#PBS -e logs/gen_fungal_comparison.err
# De novo (unconstrained) generation from the pretrained eukaryote checkpoint
# and the fungal-SFT checkpoint, for before/after evaluator comparison.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"

cd "${MRNA_GPT_ROOT}"
OUT=${MRNA_GPT_RUNS}/fungal_sft/generation
mkdir -p "$OUT"

python -m mrnagpt.generate \
    --ckpt ${MRNA_GPT_RUNS}/eukaryote/model_best.pt \
    --n 500 --out "$OUT/pretrained_eukaryote.fasta" --report "$OUT/pretrained_eukaryote_report.md"

python -m mrnagpt.generate \
    --ckpt ${MRNA_GPT_RUNS}/fungal_sft/model_best.pt \
    --n 500 --out "$OUT/fungal_sft.fasta" --report "$OUT/fungal_sft_report.md"
