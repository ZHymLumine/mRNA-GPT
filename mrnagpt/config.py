"""Config dataclasses plus YAML loading with CLI overrides."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, fields

import yaml

from .model import GPTConfig


@dataclass
class TrainConfig:
    # data
    domain: str = "archaea"
    data_dir: str = ""
    out_dir: str = ""
    # Fine-tuning: path to a pretrained model_best.pt/ckpt_*.pt to initialise
    # weights from when starting a FRESH run (no ckpt_last.pt yet in out_dir).
    # Only the model weights are loaded -- optimizer state, epoch/step counters
    # and best_val_loss all start over, which is the standard "fine-tune from a
    # checkpoint" pattern and distinct from resuming an interrupted SFT run
    # (which still resumes from this run's own ckpt_last.pt as usual).
    init_ckpt: str = ""
    train_lmdb: str = "train_codon.lmdb"
    val_lmdb: str = "val_small.lmdb"
    val_full_lmdb: str = "val_codon.lmdb"
    num_workers: int = 8

    # batching
    token_budget: int = 32768          # per-GPU micro-batch budget, in tokens
    grad_accum: int = 1
    seed: int = 42

    # optimisation
    max_epochs: int = 10
    lr: float = 2.5e-4
    min_lr: float = 2.5e-5
    schedule: str = "cosine"           # "cosine" | "wsd"
    warmup_steps: int = 750
    wsd_decay_frac: float = 0.10
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # evaluation / stopping
    eval_interval: int = 250
    eval_seqs: int = 50000             # frozen subset of val for periodic eval
    train_probe_seqs: int = 50000      # frozen slice of *train*, eval mode
    log_interval: int = 10
    early_stop_patience: int = 12
    early_stop_min_delta: float = 0.002
    early_stop_burn_in_frac: float = 0.25
    min_epochs: int = 1

    # robustness
    # 0 disables the relative-outlier test and skips only non-finite gradients.
    # clip_grad_norm_ already renormalises to grad_clip, so a norm of 5 is
    # harmless; the genuine hazard is NaN/Inf.  The relative test cost bacteria
    # 228 dead steps before being made non-latching, and it earns its keep only
    # if a run actually diverges.
    grad_norm_skip_factor: float = 0.0
    max_consecutive_skips: int = 10
    max_hours: float = 23.0
    keep_last: int = 3

    # system
    compile: bool = True
    dtype: str = "bfloat16"
    backend: str = "nccl"
    max_steps: int = 0                 # 0 = run the full epoch budget (debug knob)

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce(cur, val):
    if isinstance(cur, bool):
        return str(val).lower() in ("1", "true", "yes", "on")
    return type(cur)(val)


def load_config(argv=None) -> tuple[TrainConfig, GPTConfig, dict]:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[],
                    help="key=value pairs, e.g. lr=3e-4 model.n_layer=4")
    known, extra = ap.parse_known_args(argv)

    raw = yaml.safe_load(open(known.config)) or {}
    model_raw = raw.pop("model", {}) or {}

    # --key value  and  --key=value  both work, for PBS convenience
    pairs = list(known.override)
    i = 0
    while i < len(extra):
        tok = extra[i]
        if tok.startswith("--"):
            if "=" in tok:
                pairs.append(tok[2:])
                i += 1
            else:
                pairs.append(f"{tok[2:]}={extra[i + 1]}")
                i += 2
        else:
            i += 1

    tnames = {f.name for f in fields(TrainConfig)}
    mnames = {f.name for f in fields(GPTConfig)}
    for p in pairs:
        key, _, val = p.partition("=")
        if key.startswith("model."):
            model_raw[key[6:]] = val
        elif key in mnames and key not in tnames:
            model_raw[key] = val
        else:
            raw[key] = val

    tcfg = TrainConfig()
    for k, v in raw.items():
        if k not in tnames:
            raise KeyError(f"unknown train config key {k!r}")
        setattr(tcfg, k, _coerce(getattr(tcfg, k), v))

    mcfg = GPTConfig()
    for k, v in model_raw.items():
        if k not in mnames:
            raise KeyError(f"unknown model config key {k!r}")
        setattr(mcfg, k, _coerce(getattr(mcfg, k), v))
    mcfg.__post_init__()

    return tcfg, mcfg, {"train": tcfg.to_dict(), "model": mcfg.to_dict()}


def dump_config(resolved: dict, path: str) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=False)
