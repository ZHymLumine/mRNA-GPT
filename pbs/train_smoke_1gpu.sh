#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=03:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_smoke
#PBS -o logs/train_smoke.out
#PBS -e logs/train_smoke.err
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

source "${MRNA_GPT_ROOT}"/pbs/_train_common.sh

OUT=${MRNA_GPT_RUNS}/smoke
mkdir -p "$OUT"
DATA=$(stage_data archaea)

echo "===== A-D: preflight (env, data path, memory sweep, compile warmup)"
"$PY" tools/preflight.py --data-dir "$DATA" --out "$OUT/preflight.md" --budgets 32768,65536

echo "===== E: 2000-step loss trajectory with production hyperparameters"
rm -f "$OUT/ckpt_last.pt" "$OUT/DONE" "$OUT/RESUBMIT"
$TORCHRUN --standalone --nproc_per_node=1 -m mrnagpt.train \
    --config configs/smoke.yaml --override data_dir="$DATA" out_dir="$OUT"

echo "===== F: constrained decoding round trip"
"$PY" -m mrnagpt.generate --ckpt "$OUT/ckpt_best.pt" \
    --proteins tests/data/proteins.txt --report "$OUT/constrained_qc.md" \
    --out "$OUT/constrained.fasta" --temperature 0.8 --top-p 0.95
"$PY" -m mrnagpt.generate --ckpt "$OUT/ckpt_best.pt" --n 64 \
    --report "$OUT/unconstrained_qc.md" --out "$OUT/unconstrained.fasta" \
    --temperature 0.9 --top-p 0.95
echo "===== smoke test complete"
