#!/bin/bash
# Full data pipeline for one domain.
#
#   01 extract/filter/exact-dedup -> 02 MMseqs2 clustering -> 03 whole-cluster
#   90/10 split -> 09 val purification -> 04 codon text -> 05 LMDB
#   -> 06 leakage report -> 07 LMDB verification -> 08 QC statistics
#
# Usage:
#   DOMAIN=archaea   THREADS=32  bash scripts/run_domain.sh
#   DOMAIN=bacteria  THREADS=192 bash scripts/run_domain.sh
#   STEPS=05,06 DOMAIN=archaea bash scripts/run_domain.sh     # run only these steps
#   PURIFY=0 DOMAIN=archaea bash scripts/run_domain.sh        # skip purification, keep the raw 90/10 cluster split
set -euo pipefail

DOMAIN="${DOMAIN:?set DOMAIN=archaea|bacteria|eukaryote}"
THREADS="${THREADS:-32}"
STEPS="${STEPS:-01,02,03,09,04,05,06,07,08}"
N_QUERY="${N_QUERY:-100000}"
PURIFY="${PURIFY:-1}"
MAX_VAL="${MAX_VAL:-500000}"
# Memory cap for the mmseqs prefilter.  Each rt_HC job gets a cgroup of only
# ~320 GB (the node's 2 TB of physical memory is for the whole node; what free -g
# shows is not the quota), so without this value step 09 for bacteria is
# OOM-killed.
SPLIT_MEM="${SPLIT_MEM:-150G}"

# Clustering algorithm and parameters: cascaded cluster + connected components
# (cluster-mode 1) by default.  Do not use linclust -- measured on archaea, a
# split built from it still leaked 65% of val at >=50% homology (see README).
ALGO="${ALGO:-cluster}"
CLUSTER_MODE="${CLUSTER_MODE:-1}"
SENS="${SENS:-7.5}"
MIN_SEQ_ID="${MIN_SEQ_ID:-0.5}"
COV="${COV:-0.8}"

# Paths are configurable; defaults are relative to this repository.  MRNA_GPT_RAW
# must point at the downloaded genome archive (see README for how to obtain it).
: "${MRNA_GPT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${MRNA_GPT_DATA:=$MRNA_GPT_ROOT/data}"
: "${MRNA_GPT_RAW:=$MRNA_GPT_ROOT/raw_data}"
: "${MRNA_GPT_VOCAB:=$MRNA_GPT_ROOT/mrnagpt/vocab.txt}"
: "${MRNA_GPT_SCRATCH:=${PBS_LOCALDIR:-${TMPDIR:-/tmp}}/mrnagpt}"

P="$MRNA_GPT_ROOT"
RAW="$MRNA_GPT_RAW"
VOCAB="$MRNA_GPT_VOCAB"
D="$MRNA_GPT_DATA/$DOMAIN"
SCRATCH="${LOCAL_SCRATCH:-$MRNA_GPT_SCRATCH}"

# In raw_data the archaea directory is misspelled "archea"; the species tables
# meanwhile use archaea/eukaryota.
case "$DOMAIN" in
    archaea)   RAW_SUB=archea;    CSV=filtered_archaea_species_updated.csv ;;
    bacteria)  RAW_SUB=bacteria;  CSV=filtered_bacteria_species_updated_final.csv ;;
    eukaryote) RAW_SUB=eukaryote; CSV=filtered_eukaryota_species_updated.csv ;;
    *) echo "unknown DOMAIN=$DOMAIN" >&2; exit 1 ;;
esac

# Step 04 onwards all read this split file.
if [ "$PURIFY" = "1" ]; then SPLIT="$D/split_purified.tsv.gz"; else SPLIT="$D/split.tsv.gz"; fi

has() { [[ ",$STEPS," == *",$1,"* ]]; }

source ~/.bashrc
conda activate mrnagpt
cd "$P"
mkdir -p "$D" logs reports

if has 01; then
    echo "===== [$DOMAIN] 01 extract + dedup ====="
    rm -rf "$SCRATCH/extract/$DOMAIN"; mkdir -p "$SCRATCH/extract/$DOMAIN"
    python scripts/01_extract_cds.py --domain "$DOMAIN" \
        --raw-dir "$RAW/$RAW_SUB" --species-csv "$RAW/$CSV" \
        --out-dir "$D" --tmp-dir "$SCRATCH/extract/$DOMAIN" \
        --procs "$THREADS" --report "reports/${DOMAIN}_01_extract.md"
    rm -rf "$SCRATCH/extract/$DOMAIN"
fi

if has 02; then
    echo "===== [$DOMAIN] 02 mmseqs clustering ====="
    DOMAIN="$DOMAIN" THREADS="$THREADS" ALGO="$ALGO" CLUSTER_MODE="$CLUSTER_MODE" \
        SENS="$SENS" MIN_SEQ_ID="$MIN_SEQ_ID" COV="$COV" SPLIT_MEM="$SPLIT_MEM" \
        bash scripts/02_linclust.sh
fi

if has 03; then
    echo "===== [$DOMAIN] 03 cluster-aware split ====="
    python scripts/03_split_by_cluster.py --domain "$DOMAIN" \
        --clusters "$D/clusters.tsv.gz" --meta "$D/meta.tsv.gz" \
        --out "$D/split.tsv.gz" --report "reports/${DOMAIN}_split_stats.md"
fi

if has 09 && [ "$PURIFY" = "1" ]; then
    echo "===== [$DOMAIN] 09 purify validation ====="
    rm -rf "$SCRATCH/purify/$DOMAIN"; mkdir -p "$SCRATCH/purify/$DOMAIN"
    python scripts/09_purify_val.py --domain "$DOMAIN" \
        --cds "$D/cds.fasta.gz" --split "$D/split.tsv.gz" \
        --out "$D/split_purified.tsv.gz" \
        --work "$SCRATCH/purify/$DOMAIN" --max-val "$MAX_VAL" \
        --min-seq-id "$MIN_SEQ_ID" --cov "$COV" --sens "$SENS" \
        --split-memory-limit "$SPLIT_MEM" \
        --threads "$THREADS" --report "reports/${DOMAIN}_09_purify.md"
    rm -rf "$SCRATCH/purify/$DOMAIN"
fi

if has 04; then
    echo "===== [$DOMAIN] 04 codon text ====="
    python scripts/04_write_codon_txt.py --domain "$DOMAIN" \
        --cds "$D/cds.fasta.gz" --split "$SPLIT" --out-dir "$D" \
        --tmp-dir "$SCRATCH/tmp/$DOMAIN" --threads "$THREADS" \
        --report "reports/${DOMAIN}_04_codon_txt.md"
fi

if has 05; then
    echo "===== [$DOMAIN] 05 LMDB ====="
    python scripts/05_save_lmdb_presplit.py --domain "$DOMAIN" \
        --train-txt "$D/train_codon.txt.gz" --val-txt "$D/val_codon.txt.gz" \
        --out-dir "$D" --vocab "$VOCAB" \
        --report "reports/${DOMAIN}_05_lmdb.md"
fi

if has 06; then
    echo "===== [$DOMAIN] 06 leakage report ====="
    rm -rf "$SCRATCH/leak/$DOMAIN"; mkdir -p "$SCRATCH/leak/$DOMAIN"
    python scripts/06_leakage_report.py --domain "$DOMAIN" \
        --cds "$D/cds.fasta.gz" --split "$SPLIT" --meta "$D/meta.tsv.gz" \
        --work "$SCRATCH/leak/$DOMAIN" --out-dir reports \
        --n-query "$N_QUERY" --threads "$THREADS" --split-memory-limit "$SPLIT_MEM"
    rm -rf "$SCRATCH/leak/$DOMAIN"
fi

if has 07; then
    echo "===== [$DOMAIN] 07 verify LMDB ====="
    for s in train val; do
        python scripts/07_verify_lmdb.py \
            --lmdb "$D/${s}_codon.lmdb" --txt "$D/${s}_codon.txt.gz" \
            --vocab "$VOCAB" --n 5000
    done
fi

if has 08; then
    echo "===== [$DOMAIN] 08 QC stats ====="
    python scripts/08_qc_stats.py --domain "$DOMAIN" \
        --meta "$D/meta.tsv.gz" --split "$SPLIT" \
        --report "reports/${DOMAIN}_qc_stats.md"
fi

echo "===== [$DOMAIN] pipeline finished ====="
