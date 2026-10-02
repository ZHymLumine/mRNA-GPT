#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=06:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_icodon
#PBS -o logs/baseline_icodon.out
#PBS -e logs/baseline_icodon.err
# iCodon (Diez et al., NAR 2022): genetic-algorithm optimization of predicted
# mRNA stability over synonymous codons. Human-host, stability-objective method
# -- its predictor supports only human/mouse/fish/xenopus, there is no fungal
# model -- so it belongs with the human-host group. 10 random seeds per target
# give it a candidate set rather than a single point; each run starts from the
# native CDS adjusted to encode the panel protein exactly.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
cd "${MRNA_GPT_ROOT}"
# Uses Rscript from the active conda environment (iCodon R package required).
Rscript sft/baseline_icodon.R \
    ${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/icodon_start.fasta ${MRNA_GPT_RUNS}/fungal_sft/generation_panel/baselines/icodon.fasta 10 human 15
