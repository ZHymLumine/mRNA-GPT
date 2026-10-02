"""Atomic checkpoint save/load/prune.

All three published runs died writing a checkpoint (``PytorchStreamWriter failed
writing file data/367: file write failed``), and ``result_archea/ckpt_69000.pt``
was left truncated at 3,028,570,240 B against 3,638,109,387 B for its siblings --
unloadable, losing everything after step 62,000.  Hence: write to a temp file,
fsync, atomically rename, then verify by reloading.
"""
from __future__ import annotations

import glob
import os
import random
import re
import time

import numpy as np
import torch


def _strip_compile_prefix(sd: dict) -> dict:
    """Drop the ``_orig_mod.`` that torch.compile adds, so checkpoints are
    compile-agnostic.  Done on save, not load."""
    return {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}


def rng_states() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _as_cpu_byte(t):
    """RNG states must be CPU ByteTensors.

    Checkpoints are loaded with map_location=<cuda device> so the weights land
    where they are needed, which drags the RNG states onto the GPU too; passing
    one to set_rng_state raises "RNG state must be a torch.ByteTensor".  Resume
    had only ever been exercised on CPU, so this surfaced the first time a real
    run resumed on 8 GPUs.
    """
    return t.detach().cpu().to(torch.uint8).contiguous()


def restore_rng(states: dict | None) -> None:
    if not states:
        return
    try:
        torch.set_rng_state(_as_cpu_byte(states["torch"]))
        cu = states.get("torch_cuda")
        if cu is not None and torch.cuda.is_available():
            cu = [_as_cpu_byte(c) for c in cu]
            if len(cu) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cu)
        np.random.set_state(states["numpy"])
        random.setstate(states["python"])
    except Exception as exc:                       # never lose a run over RNG state
        print(f"[warn] could not restore RNG state ({exc}); "
              f"continuing with a fresh stream", flush=True)


def save(path: str, *, raw_model, optimizer, model_args: dict, train_cfg: dict,
         epoch: int, step_in_epoch: int, global_step: int, tokens_seen: int,
         best_val: float, wall_seconds: float, git_sha: str = "unknown",
         extra: dict | None = None, weights_only_path: str | None = None,
         verify: bool = True) -> dict:
    payload = {
        "model": _strip_compile_prefix(raw_model.state_dict()),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "model_args": model_args,
        "train_cfg": train_cfg,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val,
        "wall_seconds": wall_seconds,
        "git_sha": git_sha,
        "rng": rng_states(),
    }
    if extra:
        payload.update(extra)

    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)

    if verify:
        # a truncated write that still succeeded at the syscall level would show
        # up here rather than three weeks later
        probe = torch.load(path, map_location="cpu", weights_only=False)
        if probe["global_step"] != global_step:
            raise RuntimeError(f"checkpoint verification failed for {path}")
        del probe

    if weights_only_path:
        wtmp = weights_only_path + ".tmp"
        with open(wtmp, "wb") as fh:
            torch.save({"model": payload["model"], "model_args": model_args,
                        "global_step": global_step, "git_sha": git_sha}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(wtmp, weights_only_path)

    return {"path": path, "size_bytes": os.path.getsize(path),
            "save_time_s": round(time.time() - t0, 2)}


def load(path: str, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


_STEP_RE = re.compile(r"ckpt_step(\d+)\.pt$")


def prune(out_dir: str, keep_last: int = 3) -> list[str]:
    """Keep the newest ``keep_last`` rolling checkpoints; best/last are untouched."""
    files = sorted(glob.glob(os.path.join(out_dir, "ckpt_step*.pt")),
                   key=lambda p: int(_STEP_RE.search(p).group(1)))
    removed = []
    for p in files[:-keep_last] if keep_last > 0 else files:
        try:
            os.remove(p)
            removed.append(os.path.basename(p))
        except OSError:
            pass
    return removed
