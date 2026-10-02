#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=06:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_eval_bacexp_lr1e5
#PBS -o logs/eval_bacexp_lr1e5.out
#PBS -e logs/eval_bacexp_lr1e5.err
# Predictor-free and independent-predictor evaluation of the regenerated
# bacterial panel, plus the two tables that were missing from reports/ entirely:
#   * per-target JS/LLR against real high/low-expression TEST genes
#     (quoted in the Results but never persisted -- reports/bacexp_all_methods.md
#      carries descriptors only)
#   * the same per-target table for the stability panel at n=200, so all three
#     tasks are reported at one sample size instead of 50/200/200
# De novo descriptors for the two sweep arms are rebuilt here as well, since the
# figure pipeline reads them from evaluator_comparison.json.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate mRNAdesigner3
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=16
R=${MRNA_GPT_RUNS}
S=$R/bacexp_sweep
N=$S/generation_panel_n200
B=$N/baselines
P=$R/fungal_sft/generation_panel/baselines

# --- 1. per-target codon realism, bacterial panel -------------------------
python -m sft.evaluate_host_and_realism \
    --test-csv sft/data/bacteria_expression_test.csv \
    --ref-cai sft/lightgbm_bacexp/cai_reference.json \
    --ref-cai-label "high-expression" --alt-cai-label human --property-label expression \
    --fasta mrna_gpt_pretrained=$N/mrna_gpt_pretrained.fasta \
    --fasta mrna_gpt_sft=$N/mrna_gpt_sft.fasta \
    --fasta mrna_gpt_sft_LOW=$N/low/mrna_gpt_sft_LOW.fasta \
    --out $N/realism_per_target.json \
    --md-out reports/bacexp_panel_realism_lr1e5.md

# --- 2. per-target codon realism, stability panel at n=200 ----------------
ST=$R/stability_sft/generation_panel_n200
python -m sft.evaluate_host_and_realism \
    --test-csv sft/data/mrna_stability_test.csv \
    --ref-cai sft/lightgbm_stability/cai_reference.json \
    --ref-cai-label "high-stability" --alt-cai-label human --property-label stability \
    --fasta mrna_gpt_pretrained=$ST/mrna_gpt_pretrained.fasta \
    --fasta mrna_gpt_sft=$ST/mrna_gpt_sft.fasta \
    --fasta mrna_gpt_sft_LOW=$ST/low/mrna_gpt_sft_LOW.fasta \
    --out $ST/realism_per_target_n200.json \
    --md-out reports/stability_panel_realism_n200.md

# --- 3. all-method descriptor table, bacterial panel ----------------------
python -m sft.evaluate_methods \
    --lgbm-dir sft/lightgbm_bacexp \
    --train-csv sft/data/bacteria_expression_train.csv \
    --property-name "protein expression" \
    --fasta mrna_gpt_pretrained=$N/mrna_gpt_pretrained.fasta \
    --fasta mrna_gpt_sft=$N/mrna_gpt_sft.fasta \
    --fasta mrna_gpt_sft_LOW=$N/low/mrna_gpt_sft_LOW.fasta \
    --fasta cai_sample=$B/cai_sample_n200.fasta \
    --fasta cai_max=$B/cai_max.fasta \
    --fasta lineardesign_l0=$B/lineardesign_bacexp_l0.fasta \
    --fasta lineardesign_l1=$B/lineardesign_bacexp_l1.fasta \
    --fasta lineardesign_l4=$B/lineardesign_bacexp_l4.fasta \
    --fasta native_cds=$P/native_cds_exact.fasta \
    --fasta codongpt=$P/codongpt.fasta \
    --fasta gemorna=$P/gemorna_n200.fasta \
    --fasta icodon=$P/icodon_n200.fasta \
    --fasta codonbert=$P/codonbert_fpp_fix_ids.fasta \
    --out $N/all_methods_property_eval.json \
    --md-out reports/bacexp_all_methods_lr1e5.md

# --- 4. de novo descriptors for the two sweep arms ------------------------
python -m sft.compare_pretrained_vs_sft \
    --pretrained-fasta $R/bacexp_sft/generation/mrna_gpt_pretrained.fasta \
    --sft-fasta $S/high_lr1e5/denovo.fasta \
    --lgbm-dir sft/lightgbm_bacexp \
    --train-csv sft/data/bacteria_expression_train.csv \
    --labels mrna_gpt_pretrained,mrna_gpt_sft --property-name "protein expression" \
    --out $S/high_lr1e5/evaluator_comparison.json
python -m sft.compare_pretrained_vs_sft \
    --pretrained-fasta $R/bacexp_sft/generation/mrna_gpt_pretrained.fasta \
    --sft-fasta $S/low_lr1e5/denovo.fasta \
    --lgbm-dir sft/lightgbm_bacexp \
    --train-csv sft/data/bacteria_expression_train.csv \
    --labels mrna_gpt_pretrained,mrna_gpt_sft_LOW --property-name "protein expression" \
    --out $S/low_lr1e5/evaluator_comparison.json
