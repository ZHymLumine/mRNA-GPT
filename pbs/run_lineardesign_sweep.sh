#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=04:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_ld_sweep
#PBS -o logs/lineardesign_sweep.out
#PBS -e logs/lineardesign_sweep.err
# Finer lambda sweep so LinearDesign is compared at its own optimum rather than
# at three arbitrary points: lambda=1 already beat both 0 and 4 on distance to
# the real high-expression profile, so the interesting region was unsampled.
# Li et al. sweep 0..100; the informative range here is the low end, where CAI
# and MFE actually trade off. One process per lambda, sharing a prebuilt table.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
OUT=${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines
TABLE=$OUT/codon_usage_fungal_p75.csv
test -f "$TABLE"

for L in 0.25 0.5 0.75 1.5 2 2.5 3 5 6 8 10 15 20; do
    python -m sft.baseline_lineardesign --lambdas "$L" --tables fungal \
        --table-path "$TABLE" --meta-name "meta_fungal_l${L}.json" --out-dir "$OUT" &
done
wait
echo "sweep complete"
