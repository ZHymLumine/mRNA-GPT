#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=32
#PBS -l walltime=72:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -o logs/bacteria.out
#PBS -e logs/bacteria.err
#PBS -N mrnagpt_bacteria

# Full pretraining data pipeline for the bacteria domain.
# Each rt_HC job gets a fixed 32 cores, no GPU, and a cgroup of about 320 GB (the
# node's 2 TB of physical memory belongs to the whole node; what free -g shows is
# not the quota).  mmseqs must be kept inside that budget with
# --split-memory-limit, or it is OOM-killed.
# STEPS / MAX_VAL / N_QUERY / SPLIT_MEM can be overridden with qsub -v.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_SCRATCH:=${PBS_LOCALDIR:-${TMPDIR:-/tmp}}/mrnagpt}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc

export LOCAL_SCRATCH="$MRNA_GPT_SCRATCH"
mkdir -p "$LOCAL_SCRATCH"
echo "host=$(hostname) cpus=$(nproc) scratch=$LOCAL_SCRATCH"
echo "cgroup memory.max = $(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo unknown)"

DOMAIN=bacteria THREADS=32 \
  STEPS="${STEPS:-01,02,03,09,04,05,06,07,08}" \
  MAX_VAL="${MAX_VAL:-1000000}" N_QUERY="${N_QUERY:-50000}" \
  SPLIT_MEM="${SPLIT_MEM:-150G}" \
  bash "${MRNA_GPT_ROOT}/scripts/run_domain.sh"
