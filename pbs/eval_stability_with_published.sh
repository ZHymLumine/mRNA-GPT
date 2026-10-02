#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=08:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_stab_pub
#PBS -o logs/eval_stability_with_published.out
#PBS -e logs/eval_stability_with_published.err
# Stability panel including the three published generative/optimizer baselines.
#
# CodonGPT, GEMORNA and iCodon design from the target protein alone, so their
# sequences are identical to those used on the expression panel -- the four
# target proteins are the same file. Only the evaluator changes. They are
# reused rather than regenerated for exactly that reason.
#
# HOST NOTE, and it matters more here than on the fungal panel: the stability
# data is HUMAN mRNA half-life, and all three of these tools are human/mammalian
# (CodonGPT hardcodes a human codon table, GEMORNA's host is baked into its
# released weights, iCodon was run with --specie human). On this panel they are
# host-matched, which they were not on the fungal panel.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=16
R=${MRNA_GPT_RUNS}
N=$R/stability_sft/generation_panel_n200
B=$R/stability_sft/generation_panel/baselines
P=$R/fungal_sft/generation_panel/baselines      # the three published baselines live here
python -m sft.evaluate_methods \
    --lgbm-dir sft/lightgbm_stability \
    --train-csv sft/data/mrna_stability_train.csv \
    --property-name stability \
    --fasta mrna_gpt_pretrained=$N/mrna_gpt_pretrained.fasta \
    --fasta mrna_gpt_sft=$N/mrna_gpt_sft.fasta \
    --fasta mrna_gpt_sft_LOW=$N/low/mrna_gpt_sft_LOW.fasta \
    --fasta cai_sample=$B/cai_sample_n200.fasta \
    --fasta cai_max=$B/cai_max.fasta \
    --fasta native_cds=$P/native_cds_exact.fasta \
    --fasta codongpt=$P/codongpt.fasta \
    --fasta gemorna=$P/gemorna_n200.fasta \
    --fasta icodon=$P/icodon_n200.fasta \
    --fasta codonbert=$P/codonbert_fpp_fix_ids.fasta \
    --fasta lineardesign_l0=$B/lineardesign_stability_l0.fasta \
    --fasta lineardesign_l1=$B/lineardesign_stability_l1.fasta \
    --fasta lineardesign_l4=$B/lineardesign_stability_l4.fasta \
    --fasta lineardesign_human_l0=$B/lineardesign_human_l0.fasta \
    --fasta lineardesign_human_l1=$B/lineardesign_human_l1.fasta \
    --fasta lineardesign_human_l4=$B/lineardesign_human_l4.fasta \
    --out $N/all_methods_property_eval_full.json \
    --md-out reports/stability_all_methods_full.md
