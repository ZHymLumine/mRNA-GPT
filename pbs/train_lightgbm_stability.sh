#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1:ncpus=16
#PBS -l walltime=01:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_lgbm_stab
#PBS -o logs/train_lightgbm_stability.out
#PBS -e logs/train_lightgbm_stability.err
# Independent stability evaluator: gradient-boosted trees on codon-usage
# features, fit on the homology-clean mRNA-stability TRAIN split only
# (reports/mrna_stability_leakage.md). Never used to build the SFT set -- that
# filtering reads measured Value directly -- so it can score generations
# without circularity.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

set -euo pipefail
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate mRNAdesigner3
cd "${MRNA_GPT_ROOT}"
export OMP_NUM_THREADS=16
python -m evaluate.lightgbm_expression \
    --data-dir sft/data --prefix mrna_stability \
    --out-dir sft/lightgbm_stability
