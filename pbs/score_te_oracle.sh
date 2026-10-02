#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=02:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_score_te
#PBS -o logs/score_te_oracle.out
#PBS -e logs/score_te_oracle.err
# TE panel scored with the neural TE evaluator (held-out test Pearson 0.491 /
# Spearman 0.557 -- higher than the LightGBM TE predictor's 0.453/0.504 despite
# mRNA-LM having been pretrained on human transcripts rather than bacterial ones).
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
N=$R/te_sft/generation_panel_n200
B=$N/baselines
python sft/score_panel_translation_rate.py \
    --ckpt $R/mrna_lm_te_std/best_model.pt \
    --contexts ecoli_rbs --reference mrna_gpt_sft \
    --evaluator-note "E. coli translation efficiency evaluator: mRNA-LM fine-tuned under the standard supervised protocol on the TRAIN split of ecoli_te (folds 1-3 train / fold 4 selection / fold 5 held-out test), held-out test Pearson 0.491 / Spearman 0.557. No CAI input features. WARNING: mRNA-LM is pretrained on human transcripts, so applying it to E. coli is cross-domain transfer; the held-out test correlation above is the measure of whether that transfer holds, and it is higher than the LightGBM TE predictor without this mismatch (0.453/0.504). All records share one fixed E. coli consensus SD / terminator context, so the 5'/3' encoder inputs are constant and the signal is carried by the CDS branch." \
    --fasta mrna_gpt_pretrained=$N/mrna_gpt_pretrained.fasta \
    --fasta mrna_gpt_sft=$N/mrna_gpt_sft.fasta \
    --fasta mrna_gpt_sft_LOW=$N/low/mrna_gpt_sft_LOW.fasta \
    --fasta cai_sample=$B/cai_sample_n200.fasta \
    --fasta cai_max=$B/cai_max.fasta \
    --fasta lineardesign_l0=$B/lineardesign_te_l0.fasta \
    --fasta lineardesign_l1=$B/lineardesign_te_l1.fasta \
    --fasta lineardesign_l4=$B/lineardesign_te_l4.fasta \
    --out $N/te_oracle_scores.json \
    --md-out reports/te_panel_neural_oracle.md
