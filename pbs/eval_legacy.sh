#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=02:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_legacy
#PBS -o logs/eval_legacy.out
#PBS -e logs/eval_legacy.err
# Re-score the published checkpoints to turn the PAD-dilution correction (loss
# and perplexity excluding PAD tokens) into a measurement.  Note ckpt_69000.pt is
# NOT used: it was truncated by the disk-full failure that ended the archaea run.
# MRNA_GPT_LEGACY must point at the directory holding the originally published
# checkpoints.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

source "${MRNA_GPT_ROOT}"/pbs/_train_common.sh

: "${MRNA_GPT_LEGACY:=$MRNA_GPT_ROOT/legacy}"
OLD=${MRNA_GPT_LEGACY}
D=${MRNA_GPT_DATA}
R=${MRNA_GPT_ROOT}/reports

run() {  # domain ckpt
    local dom="$1" ck="$2"
    local val="$D/$dom/val_codon.lmdb"
    [ -f "$ck" ] || { echo "skip: no $ck"; return 0; }
    [ -f "$val.lengths.npy" ] || { echo "skip $dom: no lengths cache yet"; return 0; }
    echo "===== $dom  $(basename "$ck")"
    "$PY" tools/eval_legacy_ckpt.py --ckpt "$ck" --val-lmdb "$val" \
        --label "mRNA-GPT-$dom (published)" --max-seqs 20000 \
        --report "$R/${dom}_legacy_ckpt_reeval.md"
}

run archaea   "$OLD/result_archea/ckpt_62000.pt"
run bacteria  "$OLD/result/ckpt_563000.pt"
run eukaryote "$OLD/result_ekuaryote/ckpt_694000.pt"
