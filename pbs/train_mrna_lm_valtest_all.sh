#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=08:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_valtest_all
#PBS -o logs/train_mrna_lm_valtest_all.out
#PBS -e logs/train_mrna_lm_valtest_all.err
# One property predictor per task, all three under the SAME script and the SAME
# protocol, so Figure 5's right column is the same measurement in every row.
# Protocol: the evaluation-only pool (VAL+TEST of that property dataset) split
# into five whole-cluster folds -- 1-3 train, 4 selects the checkpoint, 5 held
# out -- so the predictor never sees the mRNA-GPT fine-tuning sequences or
# their homologs.  Previously stability and fungal used this pool but two
# different scripts, and bacteria used the standard protocol on the TRAIN split.
#
# Control: stability reuses the exact folds of the published run
# (train 5001 / select 1705 / held out 1590), so its held-out Pearson should
# come back at about 0.391.  If it does not, something other than the protocol
# changed and the other two numbers are not trustworthy either.
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
cd ${MRNA_GPT_EXTERNAL}/mRNA-LM
export MRNA_LM_WEIGHTS=${MRNA_GPT_EXTERNAL}/mRNA-LM/weights/final
export MRNA_LM_REPO=${MRNA_GPT_EXTERNAL}/mRNA-LM
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0
R=${MRNA_GPT_RUNS}

for spec in "mrna_stability:mrna_lm_stability_vt" \
            "fungal_expression:mrna_lm_fungal_vt" \
            "bacteria_expression:mrna_lm_bacexp_vt"; do
    data="${spec%%:*}"; out="${spec##*:}"
    echo "=================== $data -> $out ==================="
    python run_finetune_property.py \
        --data "data/${data}_lm_valtest.csv" \
        -o "$R/$out"
done
echo "DONE train_mrna_lm_valtest_all"
