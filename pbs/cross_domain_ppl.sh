#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=03:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_xdom_ppl
#PBS -o logs/cross_domain_ppl.out
#PBS -e logs/cross_domain_ppl.err
# Cross-domain perplexity matrix: score each domain's held-out validation split
# with each of the three pretrained models (3 x 3). If every model is best on its
# own domain, the models learned domain-specific coding patterns rather than a
# generic codon prior -- the quantitative counterpart to the embedding UMAP.
# All nine cells use the same 20,000-sequence cap so they are directly comparable.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"
: "${MRNA_GPT_DATA:=$MRNA_GPT_ROOT/data}"

set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
export PYTHONNOUSERSITE=1
OUT=${MRNA_GPT_RUNS}/_crossdomain
mkdir -p "$OUT"

for M in archaea bacteria eukaryote; do
    python tools/eval_checkpoint.py \
        --ckpt ${MRNA_GPT_RUNS}/$M/model_best.pt \
        --lmdb ${MRNA_GPT_DATA}/archaea/val_codon.lmdb   --label archaea \
        --lmdb ${MRNA_GPT_DATA}/bacteria/val_codon.lmdb  --label bacteria \
        --lmdb ${MRNA_GPT_DATA}/eukaryote/val_codon.lmdb --label eukaryote \
        --max-seqs 20000 \
        --report "$OUT/model_${M}.md"
done
