#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=01:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_smokeq
#PBS -o logs/train_smoke_quick.out
#PBS -e logs/train_smoke_quick.err
# Re-verifies only what the first smoke run could not: that the compile warmup
# now covers the training loop's actual call signature (no recompiles after
# step 0), and that the generation step runs end to end.  The 2000-step loss
# trajectory and the memory sweep are already measured.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

source "${MRNA_GPT_ROOT}"/pbs/_train_common.sh

OUT=${MRNA_GPT_RUNS}/smoke_quick
rm -rf "$OUT"; mkdir -p "$OUT"
DATA=$(stage_data archaea)

ls -l tests/data/proteins.txt   # the rsync exclude used to eat this

$TORCHRUN --standalone --nproc_per_node=1 -m mrnagpt.train \
    --config configs/smoke.yaml --override data_dir="$DATA" out_dir="$OUT" \
    max_steps=60 eval_interval=30 eval_seqs=5000 train_probe_seqs=5000

echo "===== constrained decoding round trip"
"$PY" -m mrnagpt.generate --ckpt "$OUT/ckpt_best.pt" \
    --proteins tests/data/proteins.txt --report "$OUT/constrained_qc.md" \
    --out "$OUT/constrained.fasta" --temperature 0.8 --top-p 0.95
echo "===== unconstrained"
"$PY" -m mrnagpt.generate --ckpt "$OUT/ckpt_best.pt" --n 64 \
    --report "$OUT/unconstrained_qc.md" --out "$OUT/unconstrained.fasta" \
    --temperature 0.9 --top-p 0.95
echo "===== quick smoke complete"
