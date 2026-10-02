#!/bin/bash
# 02 - MMseqs2 nucleotide clustering (50% identity / 80% coverage)
#
# Nucleotide mode is selected by createdb --dbtype 2; linclust/cluster themselves
# have no --search-type/--strand options.  We use the three explicit steps
# createdb + <algo> + createtsv rather than the easy-* wrappers: easy-linclust
# also emits _all_seqs.fasta (roughly the size of the input), which for eukaryote
# is ~170 GB of waste.  The DB and tmp dirs stay on node-local disk; only
# clusters.tsv.gz is copied back to shared storage.
#
# Note: --split-memory-limit must be passed.  The mmseqs default of 0 means "use
# all available memory", and kmermatcher pre-allocates its k-mer table in one go:
# eukaryote (91.84 M sequences / ~127 G nucleotides) was measured going straight
# to 318.7 GB, hitting the 320 GB cgroup limit of the rt_HC queue and being
# OOM-killed 35 seconds after start.
#
# ALGO=linclust  Linear-time greedy clustering that only links sequences sharing
#                an exact k-mer with the cluster representative.  Fast, but low
#                sensitivity -- measured on archaea, a split built from it still
#                left 65% of the val sequences with >=50% homology in train.
# ALGO=cluster   Cascaded clustering, internally prefilter + Smith-Waterman, with
#                sensitivity close to mmseqs search.  Combined with
#                --cluster-mode 1 (connected components / BLASTclust) it gives a
#                hard guarantee that homologues land on the same side.
#
# Usage:
#   DOMAIN=archaea ALGO=cluster CLUSTER_MODE=1 SENS=7.5 THREADS=48 TAG=_cc bash 02_linclust.sh
set -euo pipefail

DOMAIN="${DOMAIN:?set DOMAIN=archaea|bacteria|eukaryote}"
ALGO="${ALGO:-cluster}"                 # linclust | cluster
THREADS="${THREADS:-32}"
MIN_SEQ_ID="${MIN_SEQ_ID:-0.5}"
COV="${COV:-0.8}"
COV_MODE="${COV_MODE:-0}"
CLUSTER_MODE="${CLUSTER_MODE:-1}"       # 1 = connected component (BLASTclust)
SENS="${SENS:-7.5}"
KMER_PER_SEQ_SCALE="${KMER_PER_SEQ_SCALE:-0.3}"
SPLIT_MEM="${SPLIT_MEM:-150G}"          # per-split memory cap for kmermatcher / prefilter
EXTRA_ARGS="${EXTRA_ARGS:-}"
TAG="${TAG:-}"                          # when non-empty, write clusters${TAG}.tsv.gz so runs can coexist

# Paths are configurable; defaults are relative to this repository.
: "${MRNA_GPT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${MRNA_GPT_DATA:=$MRNA_GPT_ROOT/data}"
: "${MRNA_GPT_SCRATCH:=${PBS_LOCALDIR:-${TMPDIR:-/tmp}}/mrnagpt}"

P="$MRNA_GPT_ROOT"
D="$MRNA_GPT_DATA/$DOMAIN"
LOC="${LOCAL_SCRATCH:-$MRNA_GPT_SCRATCH}/mmseqs/$DOMAIN$TAG"

mkdir -p "$LOC/tmp" "$P/reports" "$P/logs"
trap 'rm -rf "$LOC"' EXIT

echo "[$DOMAIN$TAG] decompressing cds.fasta.gz -> $LOC ..."
unpigz -c "$D/cds.fasta.gz" > "$LOC/cds.fasta"
N_IN=$(grep -c '^>' "$LOC/cds.fasta")
echo "[$DOMAIN$TAG] input sequences: $N_IN"

echo "[$DOMAIN$TAG] mmseqs createdb ..."
mmseqs createdb "$LOC/cds.fasta" "$LOC/DB" --dbtype 2

if [ "$ALGO" = "linclust" ]; then
    ALGO_ARGS=(--kmer-per-seq-scale "$KMER_PER_SEQ_SCALE")
else
    ALGO_ARGS=(--cluster-mode "$CLUSTER_MODE" -s "$SENS")
fi

echo "[$DOMAIN$TAG] mmseqs $ALGO (min-seq-id=$MIN_SEQ_ID, c=$COV, ${ALGO_ARGS[*]}, threads=$THREADS) ..."
/usr/bin/time -v mmseqs "$ALGO" "$LOC/DB" "$LOC/DB_clu" "$LOC/tmp" \
    --min-seq-id "$MIN_SEQ_ID" \
    -c "$COV" --cov-mode "$COV_MODE" \
    "${ALGO_ARGS[@]}" \
    --split-memory-limit "$SPLIT_MEM" \
    --threads "$THREADS" \
    --remove-tmp-files 1 \
    $EXTRA_ARGS 2>&1 | tee "$P/logs/02_${ALGO}_${DOMAIN}${TAG}.log"

echo "[$DOMAIN$TAG] mmseqs createtsv ..."
mmseqs createtsv "$LOC/DB" "$LOC/DB" "$LOC/DB_clu" "$LOC/clusters.tsv"
pigz -p "$THREADS" -c "$LOC/clusters.tsv" > "$D/clusters${TAG}.tsv.gz"

N_MEM=$(wc -l < "$LOC/clusters.tsv")
N_CLU=$(cut -f1 "$LOC/clusters.tsv" | sort -u --parallel="$THREADS" -S 20% | wc -l)
{
    echo "# 02_${ALGO} - $DOMAIN$TAG"
    echo
    echo "- date: $(date -Iseconds)   host: $(hostname)"
    echo "- mmseqs: $(mmseqs version)"
    echo "- algo: \`mmseqs $ALGO\`"
    echo "- params: --min-seq-id $MIN_SEQ_ID -c $COV --cov-mode $COV_MODE ${ALGO_ARGS[*]} --split-memory-limit $SPLIT_MEM --threads $THREADS $EXTRA_ARGS"
    echo "- input sequences: $N_IN"
    echo "- cluster members (tsv rows): $N_MEM"
    echo "- clusters (distinct representatives): $N_CLU"
    echo "- output: \`$D/clusters${TAG}.tsv.gz\`"
} > "$P/reports/${DOMAIN}${TAG}_${ALGO}.md"

cat "$P/reports/${DOMAIN}${TAG}_${ALGO}.md"
echo "[$DOMAIN$TAG] DONE"
