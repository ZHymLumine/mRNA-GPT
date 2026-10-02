#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=32
#PBS -l walltime=06:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_lengths
#PBS -o logs/build_lengths.out
#PBS -e logs/build_lengths.err
# CPU-only.  Must not run inside a GPU job: it is a full sequential scan of the
# LMDB (~42 GB for bacteria) and would repeat on every resubmit.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_DATA:=$MRNA_GPT_ROOT/data}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate mRNAdesigner3
cd "${MRNA_GPT_ROOT}"
D=${MRNA_GPT_DATA}
for dom in archaea bacteria eukaryote; do
    for f in train_codon.lmdb val_codon.lmdb val_small.lmdb; do
        [ -f "$D/$dom/$f" ] && python tools/build_lengths.py "$D/$dom/$f"
    done
done
