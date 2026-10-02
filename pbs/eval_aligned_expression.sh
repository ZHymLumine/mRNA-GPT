#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=08:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_align_expr
#PBS -o logs/eval_aligned_expression.out
#PBS -e logs/eval_aligned_expression.err
# Expression panel with every STOCHASTIC method at n=200 per target.
# iCodon is added by pbs/eval_aligned_expression_icodon.sh once its 200-seed run
# finishes; it is the one stochastic method still regenerating.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=16
R=${MRNA_GPT_RUNS}
N=$R/fungal_sft/generation_panel_n200
B=$R/fungal_sft/generation_panel/baselines
ARGS=(--fasta mrna_gpt_pretrained=$N/pretrained_eukaryote.fasta
      --fasta mrna_gpt_sft=$N/fungal_sft.fasta
      --fasta mrna_gpt_sft_LOW=$R/fungal_sft/generation_panel_low/fungal_sft_low.fasta
      --fasta cai_sample=$B/cai_sample_n200.fasta
      --fasta gemorna=$B/gemorna_n200.fasta
      --fasta codongpt=$B/codongpt.fasta
      --fasta cai_max=$R/fungal_sft/generation_panel/cai_max.fasta
      --fasta lineardesign_l0=$B/lineardesign_fungal_l0.fasta
      --fasta lineardesign_l1=$B/lineardesign_fungal_l1.fasta
      --fasta lineardesign_l4=$B/lineardesign_fungal_l4.fasta
      --fasta lineardesign_yeast_l0=$B/lineardesign_yeast_l0.fasta
      --fasta lineardesign_yeast_l1=$B/lineardesign_yeast_l1.fasta
      --fasta lineardesign_yeast_l4=$B/lineardesign_yeast_l4.fasta
      --fasta codonbert=$B/codonbert_fpp_fix_ids.fasta
      --fasta native_cds=$B/native_cds_exact.fasta)
[ -f "$B/icodon_n200.fasta" ] && ARGS+=(--fasta icodon=$B/icodon_n200.fasta)
python -m sft.evaluate_methods \
    --lgbm-dir sft/lightgbm_expression \
    --train-csv sft/data/fungal_expression_train.csv \
    --property-name expression "${ARGS[@]}" \
    --out $N/all_methods_property_eval_aligned.json \
    --md-out reports/expression_all_methods_aligned.md
