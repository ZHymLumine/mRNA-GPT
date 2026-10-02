#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=03:00:00
# Set PBS_GROUP to your own compute-allocation / project code before submitting.
#PBS -P ${PBS_GROUP:-CHANGE_ME}
#PBS -N mrnagpt_final
#PBS -o logs/finalize.out
#PBS -e logs/finalize.err
# Run once the domain models exist.  Produces the reported artefacts: generated
# sequences per domain, their CDS-validity table against the training baseline,
# and the loss/perplexity figures.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

source "${MRNA_GPT_ROOT}"/pbs/_train_common.sh

RUNS=${MRNA_GPT_RUNS}
REPO=${MRNA_GPT_ROOT}

for dom in archaea bacteria eukaryote; do
    CK="$RUNS/$dom/ckpt_best.pt"
    # DONE, not just a checkpoint: mid-training checkpoints exist from the first
    # eval onward, and scoring one would write a "final" report for a run that is
    # still going
    [ -f "$RUNS/$dom/DONE" ] || { echo "skip $dom: not finished"; continue; }
    [ -f "$CK" ] || { echo "skip $dom: no ckpt_best.pt"; continue; }
    echo "===== $dom : protein-constrained generation"
    "$PY" -m mrnagpt.generate --ckpt "$CK" \
        --proteins tests/data/proteins.txt --temperature 0.8 --top-p 0.95 \
        --out "$RUNS/$dom/constrained.fasta" --report "$RUNS/$dom/constrained_qc.md"
    echo "===== $dom : unconstrained de novo generation"
    "$PY" -m mrnagpt.generate --ckpt "$CK" --n 1000 --temperature 0.9 --top-p 0.95 --batch-size 16 --max-codons 1200 \
        --out "$RUNS/$dom/unconstrained.fasta" --report "$RUNS/$dom/unconstrained_qc.md"
done

echo "===== full-validation evaluation of each best checkpoint"
for dom in archaea bacteria eukaryote; do
    CK="$RUNS/$dom/ckpt_best.pt"
    V=${MRNA_GPT_DATA}/$dom/val_codon.lmdb
    [ -f "$RUNS/$dom/DONE" ] && [ -f "$CK" ] && [ -f "$V.lengths.npy" ] || continue
    "$PY" tools/eval_checkpoint.py --ckpt "$CK" --lmdb "$V" --label "$dom val (full)" \
        --report "$REPO/reports/${dom}_final_eval.md"
done

echo "===== figures"
"$PY" tools/plot_curves.py --out "$REPO/reports/figs" \
    --runs $RUNS/archaea $RUNS/bacteria $RUNS/eukaryote \
           $RUNS/archaea_random $RUNS/archaea_wd0 $RUNS/archaea_learned_pe \
           $RUNS/archaea_40ep 2>/dev/null || true

echo "===== finalize complete"
