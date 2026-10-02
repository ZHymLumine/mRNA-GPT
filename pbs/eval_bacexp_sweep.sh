#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=04:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_bacexp_sweep_eval
#PBS -o logs/eval_bacexp_sweep.out
#PBS -e logs/eval_bacexp_sweep.err
# De novo generation + predictor-free scoring for every sweep point.
#
# The quantity being optimised is the SEPARATION between the high arm and the
# low control, not either arm's score: a learning rate that moves both equally
# has produced a better-looking number and no more evidence.
#
# Selection is done on the VAL split and reported on TEST. Choosing the learning
# rate on the same TEST genes the result is quoted against would be exactly the
# circularity this paper argues against elsewhere.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate mRNAdesigner3
cd "${MRNA_GPT_ROOT}"
R=${MRNA_GPT_RUNS}
S=$R/bacexp_sweep

for lr in 1e5 3e5 1e4 3e4; do
  for arm in high low; do
    CK=$S/${arm}_lr${lr}/model_best.pt
    FA=$S/${arm}_lr${lr}/denovo.fasta
    [ -f "$CK" ] || { echo "missing $CK, skipping"; continue; }
    [ -s "$FA" ] || python -m mrnagpt.generate --ckpt "$CK" --n 500 \
        --out "$FA" --report "$S/${arm}_lr${lr}/denovo_qc.md"
  done
done

# One realism report per split; every sweep point is an arm inside it, so the
# real-gene reference rows are computed once and shared across all of them.
for split in val test; do
  ARGS=(--ungrouped --test-csv sft/data/bacteria_expression_${split}.csv
        --ref-cai sft/lightgbm_bacexp/cai_reference.json
        --ref-cai-label "high-expression" --alt-cai-label human --property-label expression)
  for lr in 1e5 3e5 1e4 3e4; do
    for arm in high low; do
      FA=$S/${arm}_lr${lr}/denovo.fasta
      [ -s "$FA" ] && ARGS+=(--fasta "${arm}_lr${lr}=$FA")
    done
  done
  python -m sft.evaluate_host_and_realism "${ARGS[@]}" \
      --out "$S/sweep_realism_${split}.json" \
      --md-out "reports/bacexp_sweep_realism_${split}.md"
done

python sft/summarize_bacexp_sweep.py --sweep "$S" \
    --out reports/bacexp_sweep_summary.md
