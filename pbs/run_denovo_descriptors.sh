#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=32
#PBS -l walltime=03:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_denovo_desc
#PBS -o logs/denovo_desc.out
#PBS -e logs/denovo_desc.err
# Fills the two holes in the de novo descriptor panel (Figure 3e-h):
#   * fungal expression: the LOW negative control was generated
#     (generation_low/) but never scored for CAI/tAI/GC3/MFE.
#   * bacterial expression: none of the three arms were scored at all.
# The constrained panel already has all of these; this is the unconstrained
# arm, which is the one that shows whether fine-tuning moved the model's own
# codon preferences rather than just its choices under a protein constraint.
#
# Walltime from measurement, not guesswork: RNAfold is O(n^3) and these are
# short (fungal ~1.0 kb, bacexp 0.79-0.94 kb mean), so 500 sequences runs far
# faster than the 6 kb stability batches that needed 27 min each. 3 h is ~6x
# the expected cost; rt_HC is CPU-only so the margin is cheap.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate mRNAdesigner3
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=32
RUNS=${MRNA_GPT_RUNS}

python -m sft.compare_pretrained_vs_sft \
    --pretrained-fasta "$RUNS/fungal_sft/generation/pretrained_eukaryote.fasta" \
    --sft-fasta "$RUNS/fungal_sft/generation_low/fungal_sft_low.fasta" \
    --labels pretrained_eukaryote,fungal_sft_low \
    --property-name "expression level" --lgbm-dir sft/lightgbm_expression \
    --train-csv sft/data/fungal_expression_train.csv \
    --out "$RUNS/fungal_sft/generation/evaluator_comparison_low.json"

python -m sft.compare_pretrained_vs_sft \
    --pretrained-fasta "$RUNS/bacexp_sft/generation/mrna_gpt_pretrained.fasta" \
    --sft-fasta "$RUNS/bacexp_sft/generation/mrna_gpt_sft.fasta" \
    --labels mrna_gpt_pretrained,mrna_gpt_sft \
    --property-name "expression level" --lgbm-dir sft/lightgbm_bacexp \
    --train-csv sft/data/bacteria_expression_train.csv \
    --out "$RUNS/bacexp_sft/generation/evaluator_comparison.json"

python -m sft.compare_pretrained_vs_sft \
    --pretrained-fasta "$RUNS/bacexp_sft/generation/mrna_gpt_pretrained.fasta" \
    --sft-fasta "$RUNS/bacexp_sft/generation/mrna_gpt_sft_LOW.fasta" \
    --labels mrna_gpt_pretrained,mrna_gpt_sft_LOW \
    --property-name "expression level" --lgbm-dir sft/lightgbm_bacexp \
    --train-csv sft/data/bacteria_expression_train.csv \
    --out "$RUNS/bacexp_sft/generation/evaluator_comparison_low.json"
