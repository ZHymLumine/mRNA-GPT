#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=04:00:00
# 4 h, not the 2 h the fungal equivalent needed. RNAfold is O(n^3) per sequence
# and the de novo samples here run to 6 kb: measured 4.7 s at 2,406 nt, so the
# cubic tail dominates. Summing that cost over the length distribution of an
# existing 500-sequence de novo batch gives ~27 min per batch, ~2 h over the
# four compare_pretrained_vs_sft invocations -- this is 2x that, and rt_HC is
# CPU-only so the margin is cheap in points.
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_eval_stability
#PBS -o logs/eval_stability.out
#PBS -e logs/eval_stability.err
# Independent evaluation of everything pbs/generate_stability.sh produced.
#
# Two evaluator families, deliberately:
#   * compare_pretrained_vs_sft -- CAI, tAI, GC/GC3, MFE (ViennaRNA), novelty vs
#     the homology-clean stability TRAIN split, synonymous-variant diversity, and
#     the LightGBM stability predictor (sft/lightgbm_stability, fit on the same
#     clean split, never used to select the SFT set).
#   * evaluate_host_and_realism -- NO trained predictor: within-synonymous-family
#     codon usage against the real top/bottom-quartile genes of the held-out TEST
#     split. This is the one that can distinguish "moved toward stable" from
#     "moved toward this dataset's host".
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=16

RUNS=${MRNA_GPT_RUNS}
GEN=$RUNS/stability_sft/generation
PANEL=$RUNS/stability_sft/generation_panel
REPORTS=${MRNA_GPT_ROOT}/reports
TRAIN_CSV=sft/data/mrna_stability_train.csv
LGBM=sft/lightgbm_stability

# ---- 1. target-constrained panel, paired per target protein ----
python -m sft.compare_pretrained_vs_sft \
    --pretrained-fasta "$PANEL/pretrained_archaea.fasta" \
    --sft-fasta "$PANEL/stability_sft.fasta" \
    --labels pretrained_archaea,stability_sft \
    --property-name stability --lgbm-dir "$LGBM" --train-csv "$TRAIN_CSV" --per-target \
    --out "$PANEL/evaluator_comparison_per_target.json" \
    --md-out "$REPORTS/stability_target_panel_comparison.md"

# same panel, negative control against the same pretrained baseline
python -m sft.compare_pretrained_vs_sft \
    --pretrained-fasta "$PANEL/pretrained_archaea.fasta" \
    --sft-fasta "$PANEL/low/stability_sft_low.fasta" \
    --labels pretrained_archaea,stability_sft_low \
    --property-name stability --lgbm-dir "$LGBM" --train-csv "$TRAIN_CSV" --per-target \
    --out "$PANEL/evaluator_comparison_per_target_low.json" \
    --md-out "$REPORTS/stability_target_panel_comparison_low.md"

# ---- 2. de novo generation, no target protein ----
python -m sft.compare_pretrained_vs_sft \
    --pretrained-fasta "$GEN/pretrained_archaea.fasta" \
    --sft-fasta "$GEN/stability_sft.fasta" \
    --labels pretrained_archaea,stability_sft \
    --property-name stability --lgbm-dir "$LGBM" --train-csv "$TRAIN_CSV" \
    --out "$GEN/evaluator_comparison.json"

python -m sft.compare_pretrained_vs_sft \
    --pretrained-fasta "$GEN/pretrained_archaea.fasta" \
    --sft-fasta "$GEN/stability_sft_low.fasta" \
    --labels pretrained_archaea,stability_sft_low \
    --property-name stability --lgbm-dir "$LGBM" --train-csv "$TRAIN_CSV" \
    --out "$GEN/evaluator_comparison_low.json"

# ---- 3. predictor-free realism, anchored on real held-out TEST genes ----
REAL_ARGS=(--test-csv sft/data/mrna_stability_test.csv
           --ref-cai "$LGBM/cai_reference.json" --ref-cai-label "high-stability"
           --alt-cai-label human --property-label stability)

python -m sft.evaluate_host_and_realism "${REAL_ARGS[@]}" \
    --fasta pretrained_archaea="$PANEL/pretrained_archaea.fasta" \
    --fasta stability_sft="$PANEL/stability_sft.fasta" \
    --fasta stability_sft_LOW="$PANEL/low/stability_sft_low.fasta" \
    --out "$PANEL/stability_realism.json" \
    --md-out "$REPORTS/stability_panel_realism.md"

python -m sft.evaluate_host_and_realism "${REAL_ARGS[@]}" --ungrouped \
    --fasta pretrained_archaea="$GEN/pretrained_archaea.fasta" \
    --fasta stability_sft="$GEN/stability_sft.fasta" \
    --fasta stability_sft_LOW="$GEN/stability_sft_low.fasta" \
    --out "$GEN/stability_realism_unconstrained.json" \
    --md-out "$REPORTS/stability_unconstrained_realism.md"

# ---- 4. one overview page over all of the above ----
python sft/summarize_stability.py --runs "$RUNS" \
    --out "$REPORTS/stability_sft_summary.md"
