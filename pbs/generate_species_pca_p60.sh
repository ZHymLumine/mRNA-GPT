#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=02:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_species_p60
#PBS -o logs/gen_species_pca_p60.out
#PBS -e logs/gen_species_pca_p60.err
# Same as generate_species_pca.sh with a 60-codon conditioning prefix instead
# of 30, to see how much of the species signal in the generations is set by how
# much of the organism the model is shown before it starts writing.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
W=${MRNA_GPT_RUNS}/species_pca
export PREFIX_CODONS=60
for dom in archaea bacteria eukaryote; do
    python figures/species_pca/04_generate_by_species.py "$dom" 200 \
        "$W/${dom}_generated_p60.tsv"
    python figures/species_pca/06_embed.py "$dom" "$W/${dom}_generated_p60.tsv" \
        "$W/${dom}_generated_p60_emb.npz"
done
