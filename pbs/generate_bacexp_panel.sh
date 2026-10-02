#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=03:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_gen_bacexp
#PBS -o logs/gen_bacexp_panel.out
#PBS -e logs/gen_bacexp_panel.err
# Bacterial protein-expression panel: constrained (n=200 per target, matching the
# other panels) and de novo (500 per arm).
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate mRNAdesigner3
cd "${MRNA_GPT_ROOT}"
R=${MRNA_GPT_RUNS}
PRE=$R/bacteria/model_best.pt
OUT=$R/bacexp_sft/generation_panel_n200
python -m sft.generate_target_panel --csv data/protein_sequences.csv \
    --pretrained-ckpt $PRE --pretrained-label mrna_gpt_pretrained \
    --sft-ckpt $R/bacexp_sft/model_best.pt --sft-label mrna_gpt_sft \
    --n 200 --batch-size 25 --seed 42 --out-dir "$OUT"
python -m sft.generate_target_panel --csv data/protein_sequences.csv \
    --pretrained-ckpt $PRE --pretrained-label mrna_gpt_pretrained \
    --sft-ckpt $R/bacexp_sft_low/model_best.pt --sft-label mrna_gpt_sft_LOW --only sft \
    --n 200 --batch-size 25 --seed 42 --out-dir "$OUT/low"
G=$R/bacexp_sft/generation
mkdir -p "$G"
for a in "bacteria:mrna_gpt_pretrained" "bacexp_sft:mrna_gpt_sft" "bacexp_sft_low:mrna_gpt_sft_LOW"; do
    d="${a%%:*}"; lab="${a##*:}"
    ck=$R/$d/model_best.pt
    python -m mrnagpt.generate --ckpt "$ck" --n 500 \
        --out "$G/$lab.fasta" --report "$G/${lab}_qc.md"
done
