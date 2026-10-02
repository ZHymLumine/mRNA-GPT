#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=32
#PBS -l walltime=24:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -o logs/bacteria_v3.out
#PBS -e logs/bacteria_v3.err
#PBS -N mrnagpt_bac_v3

# bacteria: resume from step 04 (the outputs of 01/02/03/09 already exist on
# shared storage and are not recomputed).
#
# Note that the job snapshots run_domain.sh to node-local disk before executing
# it: bash reads a script by byte offset as it runs, so editing the source file
# while a job is running misaligns the running shell's read position and produces
# a baffling `unexpected EOF while looking for matching '"'`.  That is how v2
# died (09 had already finished, and 04-08 were lost for nothing).
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_SCRATCH:=${PBS_LOCALDIR:-${TMPDIR:-/tmp}}/mrnagpt}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc

export LOCAL_SCRATCH="$MRNA_GPT_SCRATCH"
mkdir -p "$LOCAL_SCRATCH"
echo "host=$(hostname) cpus=$(nproc) scratch=$LOCAL_SCRATCH"

SNAP="$LOCAL_SCRATCH/scripts_snapshot"
mkdir -p "$SNAP"
cp "${MRNA_GPT_ROOT}"/scripts/*.sh "$SNAP/"
echo "snapshotted run_domain.sh to $SNAP"

DOMAIN=bacteria THREADS=32 \
  STEPS="04,05,06,07,08" \
  MAX_VAL="1000000" N_QUERY="50000" SPLIT_MEM="150G" \
  bash "$SNAP/run_domain.sh"
