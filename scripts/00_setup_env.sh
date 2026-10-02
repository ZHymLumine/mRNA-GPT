#!/bin/bash
# 00 - create the mrnagpt conda environment and install MMseqs2
set -euo pipefail

source ~/.bashrc

ENV_NAME=mrnagpt
: "${MRNA_GPT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
P="$MRNA_GPT_ROOT"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    conda create -y -n "$ENV_NAME" python=3.10
fi

conda activate "$ENV_NAME"

conda install -y -c conda-forge -c bioconda mmseqs2
# transformers must be pinned to 4.46.3: 5.x reads this vocab.txt and recognises
# only 5 tokens, silently turning every codon into [UNK].
python -m pip install --quiet numpy lmdb tqdm "transformers==4.46.3" "tokenizers==0.20.3" pandas psutil matplotlib

mkdir -p "$P/reports"
{
    echo "# Environment record - $(date -Iseconds)"
    echo
    echo "## host"
    hostname
    echo
    echo "## mmseqs version"
    mmseqs version
    echo
    echo "## python"
    python -V
    echo
    echo "## conda list"
    conda list
} > "$P/reports/env.txt"

echo "OK -> $P/reports/env.txt"
