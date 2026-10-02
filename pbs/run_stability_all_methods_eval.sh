#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=02:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_eval_stab_all
#PBS -o logs/eval_stability_all.out
#PBS -e logs/eval_stability_all.err
# Predictor-free comparison across EVERY method on the stability panel:
# the three mRNA-GPT arms plus the CAI and LinearDesign baselines, scored on
# within-synonymous-family codon usage against the real top/bottom-quartile
# genes of the held-out TEST split. No trained predictor anywhere, so CAI-
# maximizing baselines cannot win by construction the way they can against
# LightGBM.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=16

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

python -m sft.evaluate_host_and_realism \
    --test-csv sft/data/mrna_stability_test.csv \
    --ref-cai sft/lightgbm_stability/cai_reference.json --ref-cai-label "high-stability" \
    --alt-cai-label human --property-label stability \
    "${ARGS[@]}" \
    --out $PANEL/stability_realism_all_methods.json \
    --md-out ${MRNA_GPT_ROOT}/reports/stability_all_methods_realism.md
