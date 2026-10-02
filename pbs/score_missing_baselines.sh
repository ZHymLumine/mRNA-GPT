#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=03:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_score_missing
#PBS -o logs/score_missing_baselines.out
#PBS -e logs/score_missing_baselines.err
# Figure 5's property-predictor column was missing the host-fixed baselines on
# the mRNA stability panel, and iCodon and the native CDS on the fungal panel.
# Those tools emit ONE design set per target protein whatever the task is, so
# the fungal-panel FASTA files are the sequences that belong in both panels --
# nothing is re-generated here, only scored.  Checkpoints are the
# validation-and-test-only evaluators the paper reports, not the _std ones, and
# each panel keeps the fixed UTR context its evaluator was trained under.
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
FB=$R/fungal_sft/generation_panel/baselines

# ---- mRNA stability: the five host-fixed baselines, plus the reference arm
#      mRNA-GPT (fine-tuned) so the Mann-Whitney comparisons have their anchor
P=$R/stability_sft/generation_panel
python sft/score_panel_translation_rate.py \
    --ckpt $R/mrna_lm_stability/best_model.pt \
    --contexts human_median --reference mrna_gpt_sft \
    --evaluator-note "Stability evaluator: LoRA fine-tuned on the VAL+TEST split (whole-cluster 5-fold, fold 5 held out, Pearson 0.391 / Spearman 0.389); it has seen neither the SFT sequences nor their homologues. Fixed human UTR context." \
    --fasta mrna_gpt_sft=${P}_n200/mrna_gpt_sft.fasta \
    --fasta codongpt=$FB/codongpt.fasta \
    --fasta gemorna=$FB/gemorna_n200.fasta \
    --fasta icodon=$FB/icodon.fasta \
    --fasta codonbert_fpp_fix=$FB/codonbert_fpp_fix_norm.fasta \
    --fasta native_cds=$FB/native_cds.fasta \
    --out $P/stability_oracle_missing_baselines.json \
    --md-out reports/stability_missing_baselines.md

# ---- fungal transcript expression: iCodon and the native CDS
P=$R/fungal_sft/generation_panel
python sft/score_panel_translation_rate.py \
    --ckpt $R/mrna_lm_fungal/best_model.pt \
    --contexts adh1_yeast --reference mrna_gpt_sft \
    --evaluator-note "Fungal expression evaluator: LoRA fine-tuned on the VAL+TEST split (whole-cluster 5-fold, fold 5 held out, Pearson 0.620 / Spearman 0.636). Fixed yeast ADH1 UTR context." \
    --fasta mrna_gpt_sft=${P}_n200/fungal_sft.fasta \
    --fasta icodon=$FB/icodon.fasta \
    --fasta native_cds=$FB/native_cds.fasta \
    --out $P/expression_oracle_missing_baselines.json \
    --md-out reports/expression_missing_baselines.md

echo "DONE score_missing_baselines"
