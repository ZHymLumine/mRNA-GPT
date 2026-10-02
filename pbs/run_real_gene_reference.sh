#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=32
#PBS -l walltime=04:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_real_ref
#PBS -o logs/real_ref.out
#PBS -e logs/real_ref.err
# Recompute the real-gene reference over the ENTIRE top/bottom quartile of each
# TEST split, with a bootstrap CI on the high-minus-low difference.
#
# This replaces an earlier 200-gene sample. The sample was not good enough: for
# bacterial expression the high-vs-low CAI difference is about -0.001, and
# re-drawing 200 genes moved it by more than that, so the SIGN of the headline
# number was a sampling artifact. Over the whole quartile the difference is
# -0.0006 with CI [-0.0071, +0.0064] -- it does not separate, which is the
# claim we can actually make. The GC3 inversion survives (-0.053, CI excludes
# zero) and is a real property of this dataset.
#
# RNAfold is the cost: 1,028 x ~1.35 kb for stability is the largest block.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate mRNAdesigner3
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=32
python sft/real_gene_reference.py \
    --out figures/real_gene_metrics.json \
    --fasta-dir ${MRNA_GPT_RUNS}/real_gene_panels \
    --threads 32 --boot 2000
