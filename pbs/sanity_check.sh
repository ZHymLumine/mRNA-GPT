#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=01:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_sanity
#PBS -o logs/sanity_check.out
#PBS -e logs/sanity_check.err
# Sanity-checks each released checkpoint against an untrained model of the same
# architecture and an order-0 unigram baseline, plus top-1 next-codon accuracy.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

source "${MRNA_GPT_ROOT}"/pbs/_train_common.sh
for dom in archaea bacteria eukaryote; do
    "$PY" tools/sanity_check.py --domain "$dom" --max-seqs 20000
done
