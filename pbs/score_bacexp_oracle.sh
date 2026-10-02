#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=02:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_score_bacexp
#PBS -o logs/score_bacexp_oracle.out
#PBS -e logs/score_bacexp_oracle.err
# Bacterial protein-expression panel scored by its neural evaluator
# (held-out test Pearson 0.295 / Spearman 0.311 -- essentially identical to the
# LightGBM predictor's 0.293/0.307, from a completely different model class and
# feature set, which is why 0.29 should be read as this dataset's ceiling rather
# than a shortcoming of either evaluator).
#
# The context is the same fixed E. coli SD/terminator pair the evaluator was
# trained under; any other UTR pair would be off-distribution for it.
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
cd "${MRNA_GPT_ROOT}"
export MRNA_LM_WEIGHTS=${MRNA_GPT_EXTERNAL}/mRNA-LM/weights/final
export MRNA_LM_REPO=${MRNA_GPT_EXTERNAL}/mRNA-LM
export TOKENIZERS_PARALLELISM=true
R=${MRNA_GPT_RUNS}
N=$R/bacexp_sft/generation_panel_n200
B=$N/baselines
P=$R/fungal_sft/generation_panel/baselines
python sft/score_panel_translation_rate.py \
    --ckpt $R/mrna_lm_bacexp_std/best_model.pt \
    --contexts ecoli_rbs --reference mrna_gpt_sft \
    --evaluator-note "Bacterial protein expression evaluator: mRNA-LM under the standard supervised protocol (train on TRAIN / select on VAL / report on TEST), held-out test Pearson 0.295 / Spearman 0.311, with no CAI input features. Almost identical to the LightGBM predictor on the same split (0.293/0.307) -- two model classes and two feature sets hitting the same ceiling, which says 0.29 is the information limit of this dataset. WARNING: CodonGPT / GEMORNA / iCodon / CodonBERT are human/mammalian-host tools, so using them for a bacterial property is a host mismatch and their scores mostly reflect that mismatch rather than design quality." \
    --fasta mrna_gpt_pretrained=$N/mrna_gpt_pretrained.fasta \
    --fasta mrna_gpt_sft=$N/mrna_gpt_sft.fasta \
    --fasta mrna_gpt_sft_LOW=$N/low/mrna_gpt_sft_LOW.fasta \
    --fasta cai_sample=$B/cai_sample_n200.fasta \
    --fasta cai_max=$B/cai_max.fasta \
    --fasta lineardesign_l0=$B/lineardesign_bacexp_l0.fasta \
    --fasta lineardesign_l1=$B/lineardesign_bacexp_l1.fasta \
    --fasta lineardesign_l4=$B/lineardesign_bacexp_l4.fasta \
    --fasta codongpt=$P/codongpt.fasta \
    --fasta gemorna=$P/gemorna_n200.fasta \
    --fasta icodon=$P/icodon_n200.fasta \
    --fasta codonbert=$P/codonbert_fpp_fix_ids.fasta \
    --out $N/bacexp_oracle_scores.json \
    --md-out reports/bacexp_panel_neural_oracle.md
