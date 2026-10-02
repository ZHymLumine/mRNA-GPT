#!/usr/bin/env python
"""Export a training checkpoint into a Hugging Face-ready model directory.

Training checkpoints carry the AdamW moments and the RNG state, which makes them
roughly 3.6 GB for a 303M-parameter model.  Nothing downstream of training needs
either, so what gets published is weights only, in bf16 safetensors (~600 MB).

Two checkpoint layouts are understood:

``preprint``  the earlier codebase: 69-token vocabulary, learned positional
              embeddings, ``block_size`` 1024, and an ``iter_num`` counter.  A
              stale top-level ``wte.weight`` alias of ``transformer.wte.weight``
              is dropped.
``current``   this codebase: 68-token vocabulary, RoPE, ``block_size`` 2048,
              ``global_step``/``epoch`` counters and a ``git_sha``.

The two are *not* interchangeable -- a preprint checkpoint indexes its embedding
table by the old token ids -- so the vocabulary and positional encoding are
written into ``config.json`` and the lineage is recorded in ``provenance.json``.

Usage:
    python export_for_hf.py --ckpt runs/bacteria/model_best.pt \\
                            --out export/mRNA-GPT-bacteria
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os

import torch

CODONS = ["".join(c) for c in itertools.product("ACGU", repeat=3)]
PREPRINT_SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
CURRENT_SPECIALS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _detect_lineage(state: dict) -> str:
    return "preprint" if "iter_num" in state and "global_step" not in state else "current"


def _clean_state_dict(raw: dict) -> dict:
    """Strip torch.compile prefixes and tied-weight aliases.

    safetensors refuses to serialise two names backed by the same storage, which
    is exactly what weight tying produces, so the alias is dropped rather than
    duplicated: loading code re-ties it from ``tie_weights`` in config.json.
    """
    out, seen = {}, {}
    for k, v in raw.items():
        if not torch.is_tensor(v):
            continue
        k = k[10:] if k.startswith("_orig_mod.") else k
        if k == "wte.weight":            # preprint-era top-level alias
            continue
        key = (v.data_ptr(), tuple(v.shape))
        if key in seen:                  # tied weight: keep the first name only
            print(f"  tied: {k!r} -> {seen[key]!r} (not serialised)")
            continue
        seen[key] = k
        out[k] = v.detach().cpu().contiguous()
    return out


def export(ckpt_path: str, out_dir: str, *, dtype: str = "bf16",
           lineage: str | None = None) -> dict:
    print(f"loading {ckpt_path}")
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lineage = lineage or _detect_lineage(st)
    margs = dict(st["model_args"])

    sd = _clean_state_dict(st["model"])
    n_params = sum(v.numel() for v in sd.values())
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[dtype]
    sd = {k: (v.to(torch_dtype) if v.is_floating_point() else v)
          for k, v in sd.items()}

    os.makedirs(out_dir, exist_ok=True)

    weights_path = os.path.join(out_dir, "model.safetensors")
    try:
        from safetensors.torch import save_file
        save_file(sd, weights_path, metadata={"format": "pt"})
    except ImportError:
        weights_path = os.path.join(out_dir, "pytorch_model.bin")
        torch.save(sd, weights_path)
        print("  safetensors not installed; wrote a .bin instead")

    cfg = {
        "model_type": "mrna-gpt",
        "architectures": ["GPT"],
        "lineage": lineage,
        "n_layer": margs["n_layer"],
        "n_head": margs["n_head"],
        "n_embd": margs["n_embd"],
        "block_size": margs["block_size"],
        "vocab_size": margs["vocab_size"],
        "bias": margs.get("bias", False),
        "dropout": 0.0,
        "pos_encoding": margs.get("pos_encoding", "learned"),
        "tie_weights": margs.get("tie_weights", True),
        "torch_dtype": {"bf16": "bfloat16", "fp16": "float16",
                        "fp32": "float32"}[dtype],
        "n_parameters": n_params,
    }
    for k in ("rope_theta", "rope_scaling", "rope_factor",
              "pad_token_id", "bos_token_id", "eos_token_id"):
        if k in margs:
            cfg[k] = margs[k]
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    prov = {
        "source_checkpoint": os.path.basename(ckpt_path),
        "lineage": lineage,
        "export_dtype": dtype,
        "n_parameters": n_params,
        "sha256": _sha256(weights_path),
        "size_bytes": os.path.getsize(weights_path),
    }
    for k in ("iter_num", "global_step", "epoch", "tokens_seen", "git_sha"):
        if k in st:
            prov[k] = st[k]
    best = st.get("best_val_loss")
    if best is not None:
        prov["best_val_loss_as_recorded"] = float(best)
        if lineage == "preprint":
            prov["best_val_loss_note"] = (
                "Recorded under the preprint-era convention, which averages loss "
                "over padding positions and so understates per-token loss. Not "
                "comparable to numbers reported with padding masked out."
            )
    with open(os.path.join(out_dir, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, default=str)

    specials = CURRENT_SPECIALS if lineage == "current" else PREPRINT_SPECIALS
    vocab = specials + CODONS
    if len(vocab) != cfg["vocab_size"]:
        raise RuntimeError(f"vocab size mismatch: built {len(vocab)}, "
                           f"checkpoint says {cfg['vocab_size']}")
    with open(os.path.join(out_dir, "vocab.txt"), "w") as fh:
        fh.write("\n".join(vocab) + "\n")

    print(f"  wrote {out_dir}  ({prov['size_bytes'] / 1e6:.0f} MB, "
          f"{n_params / 1e6:.1f}M params, {lineage})")
    return prov


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    ap.add_argument("--lineage", default=None, choices=("preprint", "current"),
                    help="override autodetection")
    args = ap.parse_args()
    export(args.ckpt, args.out, dtype=args.dtype, lineage=args.lineage)


if __name__ == "__main__":
    main()
