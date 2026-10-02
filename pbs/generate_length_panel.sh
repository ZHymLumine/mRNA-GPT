#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=02:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_gen_length
#PBS -o logs/gen_length_panel.out
#PBS -e logs/gen_length_panel.err
# Length-dependence analysis (cf. Duret et al. 1999):
# protein-constrained generation for 40 real fungal proteins from the held-out
# TEST split, 8 per length bin spanning 65-816 aa, 20 variants each. Fixed
# protein per target, so any trend in codon optimality with length is a property
# of the model and not of amino-acid composition drift. The pretrained checkpoint
# is generated alongside as the control: it reproduces the natural decline, and
# the question is whether fine-tuning removes it.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
python -m sft.generate_target_panel \
    --csv sft/data/length_panel_proteins.csv \
    --n 20 --batch-size 20 --seed 42 \
    --out-dir ${MRNA_GPT_RUNS}/fungal_sft/generation_length
