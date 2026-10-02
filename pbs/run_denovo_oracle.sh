#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=16
#PBS -l walltime=03:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_denovo_oracle
#PBS -o logs/denovo_oracle.out
#PBS -e logs/denovo_oracle.err
# Neural-evaluator scores for DE NOVO (unconstrained) generation.
#
# Every oracle report so far is on the protein-constrained panel; the
# unconstrained arm has never been scored by a neural evaluator. That leaves the
# de novo claim resting on codon-profile statistics and descriptors alone.
#
# Real top- and bottom-quartile genes are scored in the same run, under the same
# UTR context, so the three model arms can be read against them rather than only
# against each other. This is the only anchor available here: every baseline
# (CAI-max, LinearDesign, CodonGPT, GEMORNA, iCodon, CodonBERT) is
# protein-conditioned and cannot generate without a target protein, so there are
# no baseline arms in the unconstrained setting.
#
# Each evaluator is scored ONLY under the UTR context it was fine-tuned with --
# a different context is off-distribution and the numbers would be meaningless.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
# Relies on the conda environment that is already active (it must carry the
# mRNA-LM dependencies); set MRNA_LM_ENV to activate a named one instead.
if [ -n "${MRNA_LM_ENV:-}" ]; then conda activate "$MRNA_LM_ENV"; fi
cd "${MRNA_GPT_ROOT}"
R=${MRNA_GPT_RUNS}
P=$R/real_gene_panels

score () {                 # task ckpt context gen_dir pre sft low
    local task=$1 ckpt=$2 ctx=$3 gen=$4 pre=$5 sft=$6 low=$7
    python sft/score_panel_translation_rate.py \
        --ckpt "$R/$ckpt/best_model.pt" --contexts "$ctx" --reference mrna_gpt_sft \
        --fasta "mrna_gpt_pretrained=$gen/$pre" \
        --fasta "mrna_gpt_sft=$gen/$sft" \
        --fasta "mrna_gpt_sft_LOW=$gen/$low" \
        --fasta "REAL_high_test=$P/${task}_real_high.fasta" \
        --fasta "REAL_low_test=$P/${task}_real_low.fasta" \
        --evaluator-note "Neural evaluation of de novo generation (no target protein). Evaluator checkpoint: $ckpt, UTR context $ctx (the same one used when it was fine-tuned). REAL_high_test / REAL_low_test are the real top/bottom quartile genes of the held-out TEST split -- the entire quartile, unsampled -- scored in the same batch as anchors. WARNING: in the unconstrained setting there is no comparable baseline: CAI-max / LinearDesign / CodonGPT / GEMORNA / iCodon / CodonBERT all require a given target protein and cannot generate freely." \
        --out "$gen/denovo_neural_oracle.json" \
        --md-out "reports/${task}_denovo_neural_oracle.md"
}

score expression mrna_lm_expression_std adh1_yeast \
      $R/fungal_sft/generation \
      pretrained_eukaryote.fasta fungal_sft.fasta ../generation_low/fungal_sft_low.fasta

score stability mrna_lm_stability_std human_median \
      $R/stability_sft/generation \
      pretrained_archaea.fasta stability_sft.fasta stability_sft_low.fasta

score bacexp mrna_lm_bacexp_std ecoli_rbs \
      $R/bacexp_sft/generation \
      mrna_gpt_pretrained.fasta mrna_gpt_sft.fasta mrna_gpt_sft_LOW.fasta
