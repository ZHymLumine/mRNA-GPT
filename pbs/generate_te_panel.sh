#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=03:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_gen_te
#PBS -o logs/gen_te_panel.out
#PBS -e logs/gen_te_panel.err
# Translation-efficiency panel: both generation branches, n=200 per target to
# match the stability and expression panels.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
R=${MRNA_GPT_RUNS}
PRE=$R/bacteria/model_best.pt
OUT=$R/te_sft/generation_panel_n200
# A. protein-constrained panel
python -m sft.generate_target_panel --csv data/protein_sequences.csv \
    --pretrained-ckpt $PRE --pretrained-label mrna_gpt_pretrained \
    --sft-ckpt $R/te_sft/model_best.pt --sft-label mrna_gpt_sft \
    --n 200 --batch-size 25 --seed 42 --out-dir "$OUT"
python -m sft.generate_target_panel --csv data/protein_sequences.csv \
    --pretrained-ckpt $PRE --pretrained-label mrna_gpt_pretrained \
    --sft-ckpt $R/te_sft_low/model_best.pt --sft-label mrna_gpt_sft_LOW --only sft \
    --n 200 --batch-size 25 --seed 42 --out-dir "$OUT/low"
# B. de novo
G=$R/te_sft/generation
mkdir -p "$G"
python -m mrnagpt.generate --ckpt $PRE --n 500 \
    --out $G/mrna_gpt_pretrained.fasta --report $G/mrna_gpt_pretrained_qc.md
python -m mrnagpt.generate --ckpt $R/te_sft/model_best.pt --n 500 \
    --out $G/mrna_gpt_sft.fasta --report $G/mrna_gpt_sft_qc.md
python -m mrnagpt.generate --ckpt $R/te_sft_low/model_best.pt --n 500 \
    --out $G/mrna_gpt_sft_LOW.fasta --report $G/mrna_gpt_sft_LOW_qc.md
