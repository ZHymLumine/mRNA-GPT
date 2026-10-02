#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=01:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_gen_uncon_low
#PBS -o logs/gen_unconstrained_low.out
#PBS -e logs/gen_unconstrained_low.err
# De novo (unconstrained) generation from the NEGATIVE-CONTROL checkpoint, so the
# free-generation branch gets the same control the constrained branch has: if the
# bottom-quartile model's free generations land on the real LOW-expression codon
# profile while the top-quartile model's land on the HIGH one, the match is
# driven by the measured property and not by "any fungal fine-tune looks fungal".
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
OUT=${MRNA_GPT_RUNS}/fungal_sft/generation_low
mkdir -p "$OUT"
python -m mrnagpt.generate \
    --ckpt ${MRNA_GPT_RUNS}/fungal_sft_low/model_best.pt \
    --n 500 --out "$OUT/fungal_sft_low.fasta" --report "$OUT/fungal_sft_low_report.md"
