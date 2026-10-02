#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=02:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_utr_swap
#PBS -o logs/utr_swap_test.out
#PBS -e logs/utr_swap_test.err
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"
: "${MRNA_GPT_EXTERNAL:=$MRNA_GPT_ROOT/external}"

set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.1/12.1.1 2>/dev/null || module load cuda/12.6/12.6.1
source ~/.bashrc
# Relies on the conda environment that is already active (it must carry the
# mRNA-LM dependencies); set MRNA_LM_ENV to activate a named one instead.
if [ -n "${MRNA_LM_ENV:-}" ]; then conda activate "$MRNA_LM_ENV"; fi
cd "${MRNA_GPT_ROOT}"
export MRNA_LM_WEIGHTS=${MRNA_GPT_EXTERNAL}/mRNA-LM/weights/final
export MRNA_LM_REPO=${MRNA_GPT_EXTERNAL}/mRNA-LM
export TOKENIZERS_PARALLELISM=true
python sft/utr_swap_test.py \
    --designs mrna_gpt_sft=${MRNA_GPT_RUNS}/fungal_sft/generation_panel_n200/fungal_sft.fasta \
    --designs cai_max=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/cai_max.fasta \
    --per-target 50 \
    --out ${MRNA_GPT_RUNS}/fungal_sft/generation_panel_n200/utr_swap_test.json \
    --md-out ${MRNA_GPT_ROOT}/reports/utr_swap_test.md
