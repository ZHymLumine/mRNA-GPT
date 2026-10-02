#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=00:45:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_gen_constrained_cmp
#PBS -o logs/gen_constrained_comparison.out
#PBS -e logs/gen_constrained_comparison.err
# Protein-constrained (synonymous-codon) generation for ONE held-out fungal
# test-set protein (seq_id 2048, real measured Value=4.425, homology-clean
# test split, never seen by SFT training), from both the pretrained eukaryote
# checkpoint and the fungal-SFT checkpoint. 100 independent stochastic
# completions per model -- isolates codon-choice effects from "these are
# just different proteins" (unlike the earlier de novo comparison).
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"

cd "${MRNA_GPT_ROOT}"
OUT=${MRNA_GPT_RUNS}/fungal_sft/generation_constrained
mkdir -p "$OUT"

python -m mrnagpt.generate \
    --ckpt ${MRNA_GPT_RUNS}/eukaryote/model_best.pt \
    --proteins sft/data/target_protein_seqid2048_x100.txt \
    --out "$OUT/pretrained_eukaryote_constrained.fasta" --report "$OUT/pretrained_eukaryote_constrained_report.md"

python -m mrnagpt.generate \
    --ckpt ${MRNA_GPT_RUNS}/fungal_sft/model_best.pt \
    --proteins sft/data/target_protein_seqid2048_x100.txt \
    --out "$OUT/fungal_sft_constrained.fasta" --report "$OUT/fungal_sft_constrained_report.md"
