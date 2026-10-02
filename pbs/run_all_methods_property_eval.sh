#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=04:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_allmethods
#PBS -o logs/all_methods_property_eval.out
#PBS -e logs/all_methods_property_eval.err
# Every method on both property panels, scored by the PROPERTY PREDICTOR itself
# (LightGBM fit on that property's homology-clean train split) alongside CAI,
# tAI, GC3, MFE, protein identity, CDS validity, novelty and diversity.
#
# The predictor column is the point: CAI/tAI/MFE describe a sequence, they do not
# say whether it has the property. Both panels get the identical treatment so the
# two properties are directly comparable.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate mRNAdesigner3
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=16
R=${MRNA_GPT_RUNS}

# ---------------- mRNA stability (archaea backbone) ----------------
P=$R/stability_sft/generation_panel
B=$P/baselines
python -m sft.evaluate_methods \
    --lgbm-dir sft/lightgbm_stability \
    --train-csv sft/data/mrna_stability_train.csv \
    --property-name stability \
    --fasta mrna_gpt_pretrained=$P/pretrained_archaea.fasta \
    --fasta mrna_gpt_sft=$P/stability_sft.fasta \
    --fasta mrna_gpt_sft_LOW=$P/low/stability_sft_low.fasta \
    --fasta cai_max=$B/cai_max.fasta \
    --fasta cai_sample=$B/cai_sample.fasta \
    --fasta lineardesign_l0=$B/lineardesign_stability_l0.fasta \
    --fasta lineardesign_l1=$B/lineardesign_stability_l1.fasta \
    --fasta lineardesign_l4=$B/lineardesign_stability_l4.fasta \
    --fasta lineardesign_human_l0=$B/lineardesign_human_l0.fasta \
    --fasta lineardesign_human_l1=$B/lineardesign_human_l1.fasta \
    --fasta lineardesign_human_l4=$B/lineardesign_human_l4.fasta \
    --out $P/all_methods_property_eval.json \
    --md-out reports/stability_all_methods_property_eval.md

# ---------------- expression (eukaryote backbone) ----------------
P=$R/fungal_sft/generation_panel
B=$P/baselines
python -m sft.evaluate_methods \
    --lgbm-dir sft/lightgbm_expression \
    --train-csv sft/data/fungal_expression_train.csv \
    --property-name expression \
    --fasta mrna_gpt_pretrained=$P/pretrained_eukaryote.fasta \
    --fasta mrna_gpt_sft=$P/fungal_sft.fasta \
    --fasta mrna_gpt_sft_LOW=$R/fungal_sft/generation_panel_low/fungal_sft_low.fasta \
    --fasta cai_max=$P/cai_max.fasta \
    --fasta cai_sample=$P/cai_sample.fasta \
    --fasta lineardesign_l0=$B/lineardesign_fungal_l0.fasta \
    --fasta lineardesign_l1=$B/lineardesign_fungal_l1.fasta \
    --fasta lineardesign_l4=$B/lineardesign_fungal_l4.fasta \
    --fasta lineardesign_yeast_l0=$B/lineardesign_yeast_l0.fasta \
    --fasta lineardesign_yeast_l1=$B/lineardesign_yeast_l1.fasta \
    --fasta lineardesign_yeast_l4=$B/lineardesign_yeast_l4.fasta \
    --fasta gemorna=$B/gemorna.fasta \
    --fasta codongpt=$B/codongpt.fasta \
    --fasta icodon=$B/icodon.fasta \
    --fasta codonbert=$B/codonbert_fpp_fix.fasta \
    --fasta native_cds=$B/native_cds.fasta \
    --out $P/all_methods_property_eval.json \
    --md-out reports/expression_all_methods_property_eval.md
