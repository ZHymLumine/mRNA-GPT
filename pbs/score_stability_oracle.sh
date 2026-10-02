#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=02:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_score_stab
#PBS -o logs/score_stability_oracle.out
#PBS -e logs/score_stability_oracle.err
# Score every method on the panel with the NEURAL stability evaluator
# (mRNA-LM LoRA-fine-tuned on the mRNA-stability VAL+TEST splits only).
# Replaces the LightGBM column, which takes CAI as an input feature and
# therefore cannot fairly rank methods that shift CAI.
#
# Only the human_median context is scored: this evaluator was trained with that
# context held constant, so any other UTR pair is off-distribution for it.
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

PANEL=${MRNA_GPT_RUNS}/stability_sft/generation_panel
B=$PANEL/baselines
ARGS=(--fasta pretrained_archaea=$PANEL/pretrained_archaea.fasta
      --fasta stability_sft=$PANEL/stability_sft.fasta
      --fasta stability_sft_LOW=$PANEL/low/stability_sft_low.fasta
      --fasta cai_max=$B/cai_max.fasta
      --fasta cai_sample=$B/cai_sample.fasta)
for f in "$B"/lineardesign_*.fasta; do
    [ -e "$f" ] || continue
    ARGS+=(--fasta "$(basename "$f" .fasta)=$f")
done

python sft/score_panel_translation_rate.py \
    --ckpt ${MRNA_GPT_RUNS}/mrna_lm_stability/best_model.pt \
    --contexts human_median --reference stability_sft \
    --evaluator-note "mRNA-LM LoRA fine-tuned on the **VAL+TEST split** of mRNA_Stability.csv (8,296 records, 5,622 clusters divided into 5 whole-cluster folds; folds 1-3 train, fold 4 selection, fold 5 held-out test). The SFT training subset comes from the TRAIN split, and the three splits are whole-cluster clean at 50% identity, so the evaluator has seen neither the fine-tuning sequences nor their homologues; it has no CAI input features. **The half-life head shipped with mRNA-LM was not used**: that data shares 76.9% of its CDS exactly with mRNA_Stability.csv, which would put 30.6% of our SFT training subset into the evaluator's supervised training set. All records and scores share one fixed human UTR context (ENST00000320934), so the 5'/3' encoder inputs are constant and the signal is carried by the CDS branch -- it is effectively a CDS-only predictor wearing the mRNA-LM architecture." \
    "${ARGS[@]}" \
    --out $PANEL/stability_oracle_scores.json \
    --md-out ${MRNA_GPT_ROOT}/reports/stability_panel_neural_oracle.md
