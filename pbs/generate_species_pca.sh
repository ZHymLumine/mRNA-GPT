#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=04:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_species_pca
#PBS -o logs/gen_species_pca.out
#PBS -e logs/gen_species_pca.err
# Figure 2d and Supplementary Figure S1f,g.
#
# Cluster the models' own final-hidden-state embeddings, 500 coding sequences
# per species, coloured by species, PCA then UMAP.  Note what is embedded here:
# not real coding sequences, but sequences the pretrained
# model wrote itself, conditioned on a 60-codon prefix from each species and
# with that prefix discarded.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"

set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate "${MRNA_GPT_ENV:-mrnagpt}"
cd "${MRNA_GPT_ROOT}"
W=${MRNA_GPT_RUNS}/species_pca
for dom in archaea bacteria eukaryote; do
    python figures/species_pca/04_generate_by_species.py "$dom" 500 \
        "$W/${dom}_generated_p60.tsv"
    python figures/species_pca/06_embed.py "$dom" "$W/${dom}_generated_p60.tsv" \
        "$W/${dom}_generated_p60_emb.npz"
    # the same embedding for real coding sequences of the same species: the
    # reference the generated clustering is read against
    python figures/species_pca/06_embed.py "$dom" "$W/${dom}_real.tsv" \
        "$W/${dom}_real_emb.npz"
done
