#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=03:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_codongpt
#PBS -o logs/baseline_codongpt.out
#PBS -e logs/baseline_codongpt.err
# CodonGPT (Nanil Therapeutics, NAR 2025) protein-constrained generation on the
# target panel, 200 variants per target -- same synonymous-masking mechanism as
# mrnagpt/generate.py, so it is the closest generative comparator. The cached
# decoder is verified against the shipped per-codon model.generate() reference
# under greedy decoding before sampling starts.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"
: "${MRNA_GPT_EXTERNAL:=$MRNA_GPT_ROOT/external}"

set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.1/12.1.1 2>/dev/null || module load cuda/12.6/12.6.1
source ~/.bashrc
# Relies on the conda environment that is already active (it must carry the
# mRNA-LM dependencies); set MRNA_LM_ENV to activate a named one instead.
if [ -n "${MRNA_LM_ENV:-}" ]; then conda activate "$MRNA_LM_ENV"; fi
cd "${MRNA_GPT_ROOT}"
export CODONGPT_DIR=${MRNA_GPT_EXTERNAL}/codonGPT
export TOKENIZERS_PARALLELISM=false
python sft/baseline_codongpt.py --verify-equivalence \
    --n 200 --seed 42 --out ${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/codongpt.fasta
