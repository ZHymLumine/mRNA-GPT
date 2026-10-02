# Sourced by every training job.  Three cluster facts shape this file:
#   * `qsub -v` does not work here, so parameters must live inside the script;
#   * editing a script while a job runs corrupts the running bash's byte offset
#     (that is how the bacteria_v2 data job died), so we snapshot to node-local
#     scratch and run from there -- which also makes auto-resubmit safe;
#   * compute nodes have no outbound internet, so wandb stays offline.
# On the original cluster, compute and storage were billed to separate project
# allocations deliberately: one project paid for CPU/GPU hours (PBS bills jobs,
# while file writes are authorised by unix group membership), a second held the
# 1 TB dataset quota, and a third the 150 TB run directory.  Set PBS_GROUP to
# your own allocation; point MRNA_GPT_DATA / MRNA_GPT_RUNS at whatever storage
# you have.
#
# Paths are configurable.  MRNA_GPT_ROOT falls back to the directory qsub was
# invoked from (PBS_O_WORKDIR), then to the parent of this file.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${MRNA_GPT_RUNS:=$MRNA_GPT_ROOT/runs}"
: "${MRNA_GPT_DATA:=$MRNA_GPT_ROOT/data}"
: "${MRNA_GPT_EXTERNAL:=$MRNA_GPT_ROOT/external}"
: "${MRNA_GPT_SCRATCH:=${PBS_LOCALDIR:-${TMPDIR:-/tmp}}/mrnagpt}"

set -euo pipefail

source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate mRNAdesigner3

REPO="$MRNA_GPT_ROOT"
SCRATCH="$MRNA_GPT_SCRATCH"
SNAP="$SCRATCH/mrnagpt_code_${PBS_JOBID%%.*}"
mkdir -p "$SNAP"
rsync -a --exclude /data --exclude /.git --exclude /logs --exclude /reports \
      --exclude '*.pdf' --exclude '*.docx' "$REPO/" "$SNAP/"
# The snapshot is not a git repo, so `git rev-parse` inside it fails and every
# checkpoint recorded "unknown".  Exact versions are needed for reproducibility,
# so capture the sha from the real repo before switching directories.
MRNAGPT_GIT_SHA="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
    MRNAGPT_GIT_SHA="${MRNAGPT_GIT_SHA}-dirty"
fi
export MRNAGPT_GIT_SHA
echo "git sha: $MRNAGPT_GIT_SHA"

cd "$SNAP"
export PYTHONPATH="$SNAP:${PYTHONPATH:-}"

# ~/.local/bin comes first on PATH here and its torchrun carries a
# `#!/usr/bin/python` (3.9) shebang, so plain `torchrun` loads the wrong torch and
# dies on ModuleNotFoundError.  Go through the env's own interpreter instead, and
# keep ~/.local/lib off sys.path so it cannot shadow conda packages either.
export PYTHONNOUSERSITE=1
PY="$(command -v python)"
TORCHRUN="$PY -m torch.distributed.run"
case "$PY" in
    */envs/mRNAdesigner3/bin/python) ;;
    *) echo "FATAL: python is $PY, expected the mRNAdesigner3 env" >&2; exit 1 ;;
esac
"$PY" - <<'PYCHK'
import sys, torch
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.startswith("2.5"), torch.__version__
print(f"python {sys.version.split()[0]} | torch {torch.__version__} | "
      f"cuda {torch.cuda.is_available()} x{torch.cuda.device_count()}")
PYCHK

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
[ -f ~/.config/mrnagpt/env ] && . ~/.config/mrnagpt/env || true

# PBS only copies the -o/-e files back when the job ends, so mirror everything to
# a live log on shared storage; that is the only way to watch a run in flight.
LIVE_DIR="$MRNA_GPT_RUNS/_live"
mkdir -p "$LIVE_DIR"
LIVE_LOG="$LIVE_DIR/${PBS_JOBNAME:-job}.${PBS_JOBID%%.*}.log"
exec > >(stdbuf -oL tee -a "$LIVE_LOG") 2>&1
echo "live log: $LIVE_LOG"

echo "host=$(hostname) cpus=$(nproc) gpus=$(nvidia-smi -L 2>/dev/null | wc -l) snap=$SNAP"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Copy the LMDB to node-local NVMe.  Bucketed batching reads randomly across the
# whole file, and measured Lustre random-get throughput on the 39 GB bacteria
# LMDB is 1,121 seq/s against the ~3,400 seq/s training needs -- with
# stripe_count=1 all 8 ranks would be hammering a single OST.
stage_data() {
    local dom="$1"
    local src="$MRNA_GPT_DATA/$dom"
    local dst="$SCRATCH/data/$dom"
    mkdir -p "$dst"
    for f in train_codon.lmdb val_small.lmdb val_codon.lmdb; do
        [ -f "$src/$f" ] || { echo "missing $src/$f" >&2; return 1; }
        for x in "$f" "$f.lengths.npy" "$f.lengths.npy.meta"; do
            [ -f "$src/$x" ] && rsync -a "$src/$x" "$dst/"
        done
    done
    df -h "$dst" | tail -1 >&2
    echo "$dst"
}

# Same script file can be qsub'ed verbatim forever: the trainer always resumes
# from ckpt_last.pt if present, and writes DONE when the budget is exhausted.
resubmit_if_needed() {
    local self="$1" out="$2"
    [ -f "$out/DONE" ] && { echo "training complete -> no resubmit"; return 0; }
    local n; n=$(cat "$out/.resubmits" 2>/dev/null || echo 0)
    # Each resubmit costs another full walltime of points at job start, so the cap
    # is deliberately low: with right-sized walltimes, needing more than a couple
    # of resubmits means something is wrong and should be looked at, not retried.
    if [ "$n" -ge "${MAX_RESUBMITS:-3}" ]; then
        echo "resubmit cap ${MAX_RESUBMITS:-3} reached -- investigate before retrying"
        return 0
    fi
    echo $((n + 1)) > "$out/.resubmits"
    qsub "$self"      # safe: we are executing from $SNAP, not from $self
}
