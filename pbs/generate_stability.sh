#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=02:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_gen_stability
#PBS -o logs/gen_stability.out
#PBS -e logs/gen_stability.err
# Both generation branches of the stability pipeline, in one job (three
# checkpoints x two modes is cheaper as one 2 h GPU job than six 1 h ones, and
# points are charged on requested walltime):
#
#   A. WITHOUT a target protein -- de novo sampling, 500 sequences each from the
#      pretrained archaea checkpoint, the high-stability SFT checkpoint and the
#      low-stability negative control.
#   B. WITH a target protein -- synonymous-codon-constrained decoding over the
#      four real targets in data/protein_sequences.csv (the same panel used for
#      the fungal work), 50 independent variants per target per checkpoint.
#      Translation equals the target by construction, so every downstream metric
#      is a paired comparison at fixed amino-acid sequence.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"

RUNS=${MRNA_GPT_RUNS}
PRE=$RUNS/archaea/model_best.pt
SFT=$RUNS/stability_sft/model_best.pt
LOW=$RUNS/stability_sft_low/model_best.pt
for f in "$PRE" "$SFT" "$LOW"; do
    [ -f "$f" ] || { echo "missing checkpoint $f -- has training finished?" >&2; exit 1; }
done

# ---- A. unconstrained ----
OUT=$RUNS/stability_sft/generation
mkdir -p "$OUT"
python -m mrnagpt.generate --ckpt "$PRE" --n 500 \
    --out "$OUT/pretrained_archaea.fasta" --report "$OUT/pretrained_archaea_report.md"
python -m mrnagpt.generate --ckpt "$SFT" --n 500 \
    --out "$OUT/stability_sft.fasta" --report "$OUT/stability_sft_report.md"
python -m mrnagpt.generate --ckpt "$LOW" --n 500 \
    --out "$OUT/stability_sft_low.fasta" --report "$OUT/stability_sft_low_report.md"

# ---- B. target-protein constrained ----
PANEL=$RUNS/stability_sft/generation_panel
python -m sft.generate_target_panel \
    --csv data/protein_sequences.csv \
    --pretrained-ckpt "$PRE" --pretrained-label pretrained_archaea \
    --sft-ckpt "$SFT" --sft-label stability_sft \
    --n 50 --batch-size 25 --seed 42 --out-dir "$PANEL"

# the negative control on the same panel; --only sft skips re-sampling the
# pretrained checkpoint, which is already in $PANEL
python -m sft.generate_target_panel \
    --csv data/protein_sequences.csv \
    --pretrained-ckpt "$PRE" --pretrained-label pretrained_archaea \
    --sft-ckpt "$LOW" --sft-label stability_sft_low --only sft \
    --n 50 --batch-size 25 --seed 42 --out-dir "$PANEL/low"
