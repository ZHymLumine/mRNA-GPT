"""Append-only JSONL metrics (the source of truth) mirrored to CSV."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time


def git_sha(repo: str | None = None) -> str:
    # PBS jobs execute from a snapshot outside the repo, so the caller exports the
    # real sha; falling through to a local `git rev-parse` there yields "unknown".
    env = os.environ.get("MRNAGPT_GIT_SHA")
    if env:
        return env
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=repo or os.getcwd(), capture_output=True, text=True)
        sha = out.stdout.strip() or "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo or os.getcwd(),
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


class MetricWriter:
    """Rank-0-only writer.  ``metrics.jsonl`` is authoritative; CSV is a mirror."""

    CSV_FIELDS = [
        "event", "wall_seconds", "epoch", "step_in_epoch", "global_step",
        "tokens_seen", "tokens_seen_padded", "lr",
        "train_loss", "train_ppl", "train_loss_incl_pad", "train_ppl_incl_pad",
        "val_loss", "val_ppl", "val_loss_incl_pad", "val_ppl_incl_pad",
        "train_probe_loss", "generalization_gap",
        "grad_norm", "grad_clipped_frac", "skipped_steps",
        "step_time_ms", "data_wait_ms", "tokens_per_s_real", "tokens_per_s_padded",
        "pad_frac", "mfu_padded", "mfu_effective",
        "gpu_mem_alloc_gb", "gpu_mem_reserved_gb", "micro_batches", "mean_bucket_T",
        "best_val_loss", "evals_since_improve", "split", "path", "note",
    ]

    def __init__(self, out_dir: str, enabled: bool = True, run_id: str | None = None):
        self.enabled = enabled
        self.t0 = time.time()
        if not enabled:
            return
        os.makedirs(out_dir, exist_ok=True)
        self.jsonl = open(os.path.join(out_dir, "metrics.jsonl"), "a", buffering=1)
        csv_path = os.path.join(out_dir, "metrics.csv")
        new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        self.csv_fh = open(csv_path, "a", newline="")
        self.csv = csv.DictWriter(self.csv_fh, fieldnames=self.CSV_FIELDS,
                                  extrasaction="ignore")
        if new:
            self.csv.writeheader()
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self.sha = git_sha()

    def log(self, event: str, **kw):
        if not self.enabled:
            return
        rec = {"event": event, "wall_clock_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "wall_seconds": round(time.time() - self.t0, 2),
               "run_id": self.run_id, "git_sha": self.sha}
        rec.update({k: v for k, v in kw.items() if v is not None})
        self.jsonl.write(json.dumps(rec) + "\n")
        self.csv.writerow(rec)
        self.csv_fh.flush()

    def close(self):
        if self.enabled:
            self.jsonl.close()
            self.csv_fh.close()


def write_env_snapshot(path: str) -> None:
    parts = [f"# environment snapshot {time.strftime('%Y-%m-%dT%H:%M:%S')}",
             f"python: {sys.version}"]
    try:
        import torch
        parts += [f"torch: {torch.__version__} cuda={torch.version.cuda}",
                  f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"]
    except Exception as exc:                                    # pragma: no cover
        parts.append(f"torch: unavailable ({exc})")
    for cmd in (["pip", "freeze"], ["nvidia-smi"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            parts += [f"\n## {' '.join(cmd)}", out.stdout]
        except Exception:
            pass
    with open(path, "w") as fh:
        fh.write("\n".join(parts))
