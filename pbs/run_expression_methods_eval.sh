#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=04:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_expr_methods
#PBS -o logs/expression_methods_eval.out
#PBS -e logs/expression_methods_eval.err
# Expression half of the all-methods property evaluation. Split out from
# pbs/run_all_methods_property_eval.sh after that job completed the stability
# half and then died on codonbert_fpp_fix.fasta, whose FASTA ids are bare target
# names rather than label|target|index; codonbert_fpp_fix_ids.fasta is the
# renamed copy.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=16
R=${MRNA_GPT_RUNS}
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
    --fasta codonbert=$B/codonbert_fpp_fix_ids.fasta \
    --fasta native_cds=$B/native_cds.fasta \
    --out $P/all_methods_property_eval.json \
    --md-out reports/expression_all_methods_property_eval.md
