#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=02:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_eval_panel
#PBS -o logs/eval_target_panel.out
#PBS -e logs/eval_target_panel.err
# Independent-evaluator comparison of the target panel, per target protein
# (--per-target): CAI, tAI, GC/GC3, MFE, novelty vs the fungal training set,
# synonymous-variant diversity, LightGBM-predicted expression.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=16
PANEL=${MRNA_GPT_RUNS}/fungal_sft/generation_panel
python -m sft.compare_pretrained_vs_sft \
    --pretrained-fasta "$PANEL/pretrained_eukaryote.fasta" \
    --sft-fasta "$PANEL/fungal_sft.fasta" \
    --per-target \
    --out "$PANEL/evaluator_comparison_per_target.json" \
    --md-out ${MRNA_GPT_ROOT}/reports/target_panel_comparison.md
