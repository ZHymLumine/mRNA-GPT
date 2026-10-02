#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=01:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_score_panel
#PBS -o logs/score_panel_mrna_lm.out
#PBS -e logs/score_panel_mrna_lm.err
# Second independent evaluator on the target panel: the published mRNA-LM model
# (LoRA-tuned once on its own bundled 5-class task, never on our fungal data),
# applied to both checkpoints' constrained generations under one fixed ADH1 UTR
# context. CDS max_length is 1024 codons -- the longest panel protein is 676, so
# nothing is truncated.
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
PANEL=${MRNA_GPT_RUNS}/fungal_sft/generation_panel
CKPT=${MRNA_GPT_RUNS}/mrna_lm_5class/best_model.pt

for LAB in pretrained_eukaryote fungal_sft; do
    python evaluate/mrna_lm_score.py --checkpoint "$CKPT" \
        --fasta "$PANEL/$LAB.fasta" --out "$PANEL/mrna_lm_$LAB.jsonl" --batch-size 8
done

python sft/aggregate_mrna_lm_panel.py \
    --pretrained-jsonl "$PANEL/mrna_lm_pretrained_eukaryote.jsonl" \
    --sft-jsonl "$PANEL/mrna_lm_fungal_sft.jsonl" \
    --out-json "$PANEL/mrna_lm_per_target.json" \
    --out-md ${MRNA_GPT_ROOT}/reports/target_panel_mrna_lm.md
