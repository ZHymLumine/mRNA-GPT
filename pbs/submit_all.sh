#!/bin/bash
# Launch order for the full retraining.  Not a PBS job -- run it from the login
# node, or just qsub the pieces yourself.  Every train_*.sh resumes from
# ckpt_last.pt unconditionally and resubmits itself until it writes DONE, so
# re-running this script is safe.
: "${MRNA_GPT_ROOT:=${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

set -euo pipefail
cd "${MRNA_GPT_ROOT}"

need_lengths() {   # $1 = domain
    [ -f "data/$1/train_codon.lmdb.lengths.npy" ] || {
        echo "missing lengths cache for $1 -- qsub pbs/build_lengths.sh first" >&2
        return 1
    }
}

case "${1:-help}" in
  archaea)     need_lengths archaea        && qsub pbs/train_archaea.sh ;;
  bacteria)    need_lengths bacteria       && qsub pbs/train_bacteria.sh ;;
  eukaryote)   need_lengths eukaryote      && qsub pbs/train_eukaryote.sh ;;
  ablations)   need_lengths archaea_random && for v in archaea_random archaea_wd0 \
                   archaea_learned_pe archaea_40ep; do qsub "pbs/train_${v}.sh"; done ;;
  smoke)       qsub pbs/train_smoke_1gpu.sh ;;
  legacy)      qsub pbs/eval_legacy.sh ;;
  *) cat <<'USAGE'
usage: bash pbs/submit_all.sh <stage>

  smoke      rt_HG single-GPU preflight + 2000-step trajectory  (~1.5 h)
  legacy     rt_HG re-score the published checkpoints           (~0.5 h)
  archaea    rt_HF 10 epochs                                    (~1.8 h)
  ablations  rt_HF random-split / wd=0 / learned-PE / 40-epoch  (~12.7 h total)
  bacteria   rt_HF 1 epoch, 53,244 steps                        (~7.9 h)
  eukaryote  rt_HF 1 epoch                                      (~10.7 h)

Recommended order: smoke -> legacy -> archaea -> ablations -> bacteria -> eukaryote.
Run archaea first: it exercises the whole 8-GPU path in under two hours.
USAGE
  ;;
esac
