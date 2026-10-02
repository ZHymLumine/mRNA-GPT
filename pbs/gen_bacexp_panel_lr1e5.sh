#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=03:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_gen_bacexp_lr1e5
#PBS -o logs/gen_bacexp_panel_lr1e5.out
#PBS -e logs/gen_bacexp_panel_lr1e5.err
# Bacterial protein-expression panel, regenerated from the sweep arms whose
# cosine schedule actually completes (configs/sweep/bacexp_{high,low}_lr1e5.yaml).
#
# Why this run and not runs/bacexp_sft: the original config sets max_epochs 60
# with a cosine schedule, but both arms early-stop around epoch 3, so the
# learning rate never decays. Under that schedule the low-property control moved
# toward the high-expression codon profile -- i.e. the negative control failed.
# Re-running with max_epochs 12 restores the expected ordering at 1e-5, 3e-5 and
# 1e-4; 1e-5 is the point selected on the VAL split (lowest VAL JS for the high
# arm with the control correctly ordered behind it) and is reported on TEST.
#
# The pretrained arm is unchanged, so its fasta is linked rather than resampled;
# the baseline designs depend only on the target proteins and are linked too.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate mRNAdesigner3
cd "${MRNA_GPT_ROOT}"
R=${MRNA_GPT_RUNS}
S=$R/bacexp_sweep
OUT=$S/generation_panel_n200
OLD=$R/bacexp_sft/generation_panel_n200
mkdir -p "$OUT"
ln -sfn "$OLD/mrna_gpt_pretrained.fasta" "$OUT/mrna_gpt_pretrained.fasta"
ln -sfn "$OLD/baselines" "$OUT/baselines"
python -m sft.generate_target_panel --csv data/protein_sequences.csv \
    --pretrained-ckpt $R/bacteria/model_best.pt --pretrained-label mrna_gpt_pretrained \
    --sft-ckpt $S/high_lr1e5/model_best.pt --sft-label mrna_gpt_sft --only sft \
    --n 200 --batch-size 25 --seed 42 --out-dir "$OUT"
python -m sft.generate_target_panel --csv data/protein_sequences.csv \
    --pretrained-ckpt $R/bacteria/model_best.pt --pretrained-label mrna_gpt_pretrained \
    --sft-ckpt $S/low_lr1e5/model_best.pt --sft-label mrna_gpt_sft_LOW --only sft \
    --n 200 --batch-size 25 --seed 42 --out-dir "$OUT/low"
