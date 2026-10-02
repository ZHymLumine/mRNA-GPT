"""Re-score a published (69-token, block_size 1024) checkpoint.

Purpose: turn the PAD-dilution correction from an estimate into a measurement,
on the *published* models, without waiting for any new training.

Scope note: this does **not** measure the leakage inflation.  The published runs
trained on a random 90% of essentially this corpus, so they have already seen most
of the homology-clean validation set; scoring them on it would be scoring them on
their own training data.  The leakage number has to come from two freshly trained
models that differ only in the split rule (see tools/make_random_split.py).
"""
from __future__ import annotations

import argparse
import os
import sys

import lmdb
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrnagpt.data import BUCKET_EDGES, build_plan, load_lengths  # noqa: E402
from mrnagpt.model import GPT, GPTConfig                          # noqa: E402


class LegacyLMDB:
    """Yields raw legacy ids (69-token vocab); no remap, since the old model's
    embedding table is indexed by the old ids."""

    def __init__(self, path):
        self.path = path
        self.env = None
        # legacy stored length is n_codon+4; the model consumes it as-is
        self.lengths = load_lengths(path) + 2

    def get(self, i):
        if self.env is None:
            self.env = lmdb.open(self.path, subdir=False, readonly=True, lock=False,
                                 readahead=False, meminit=False)
        with self.env.begin(buffers=True) as txn:
            return np.frombuffer(txn.get(b"%d" % i), dtype=np.uint8).copy()


def load_legacy(ckpt_path, device):
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ma = st["model_args"]
    cfg = GPTConfig(vocab_size=ma["vocab_size"], block_size=ma["block_size"],
                    n_layer=ma["n_layer"], n_head=ma["n_head"], n_embd=ma["n_embd"],
                    bias=ma["bias"], dropout=0.0, pos_encoding="learned")
    model = GPT(cfg)
    sd = {}
    for k, v in st["model"].items():
        k = k[10:] if k.startswith("_orig_mod.") else k
        if k == "wte.weight":          # top-level alias of transformer.wte.weight
            continue
        sd[k] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys: {unexpected[:5]}")
    if [m for m in missing if "inv_freq" not in m]:
        raise RuntimeError(f"missing keys: {missing[:5]}")
    return model.to(device).eval(), cfg, st


@torch.no_grad()
def score(model, ds, cfg, device, mode, max_seqs, seed=1234):
    """mode='bucketed' -> per-real-token; mode='pad1024' -> the published convention."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds.lengths), size=min(max_seqs, len(ds.lengths)),
                     replace=False)
    nll_real = n_real = nll_pad = n_pad = 0.0

    if mode == "pad1024":
        W = cfg.block_size
        B = max(1, 32768 // W)
        for s in range(0, len(idx), B):
            chunk = idx[s:s + B]
            x = torch.zeros(len(chunk), W, dtype=torch.long)
            y = torch.zeros(len(chunk), W, dtype=torch.long)
            for i, j in enumerate(chunk):
                ids = ds.get(int(j))[:W]
                n = len(ids)
                x[i, :n - 1] = torch.from_numpy(ids[:-1].astype(np.int64))
                y[i, :n - 1] = torch.from_numpy(ids[1:].astype(np.int64))
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                logits, _ = model(x)
            per = F.cross_entropy(logits.reshape(-1, cfg.vocab_size).float(),
                                  y.reshape(-1), reduction="none")
            m = y.reshape(-1) != 0
            nll_real += float((per * m).sum()); n_real += float(m.sum())
            nll_pad += float((per * ~m).sum()); n_pad += float((~m).sum())
    else:
        sub = ds.lengths[idx]
        plan = build_plan(np.minimum(sub, cfg.block_size), 32768, 1, 1, seed, 0)
        for m_i in range(plan.mb_start.size):
            o, b, T = (int(plan.mb_start[m_i]), int(plan.mb_size[m_i]),
                       int(plan.mb_T[m_i]))
            rows = plan.order[o:o + b]
            x = torch.zeros(b, T, dtype=torch.long)
            y = torch.zeros(b, T, dtype=torch.long)
            for i, r in enumerate(rows):
                if r < 0:
                    continue
                ids = ds.get(int(idx[r]))[:T]
                n = len(ids)
                x[i, :n - 1] = torch.from_numpy(ids[:-1].astype(np.int64))
                y[i, :n - 1] = torch.from_numpy(ids[1:].astype(np.int64))
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                logits, _ = model(x)
            per = F.cross_entropy(logits.reshape(-1, cfg.vocab_size).float(),
                                  y.reshape(-1), reduction="none")
            mk = y.reshape(-1) != 0
            nll_real += float((per * mk).sum()); n_real += float(mk.sum())
            nll_pad += float((per * ~mk).sum()); n_pad += float((~mk).sum())

    excl = nll_real / max(n_real, 1)
    incl = (nll_real + nll_pad) / max(n_real + n_pad, 1)
    return {"loss_excl_pad": excl, "ppl_excl_pad": float(np.exp(excl)),
            "loss_incl_pad": incl, "ppl_incl_pad": float(np.exp(incl)),
            "pad_frac": n_pad / max(n_real + n_pad, 1),
            "n_real_tokens": int(n_real), "n_seqs": int(len(idx))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-lmdb", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--max-seqs", type=int, default=20000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    model, cfg, st = load_legacy(args.ckpt, args.device)
    ds = LegacyLMDB(args.val_lmdb)
    label = args.label or os.path.basename(args.ckpt)

    rows = []
    for mode in ("pad1024", "bucketed"):
        r = score(model, ds, cfg, args.device, mode, args.max_seqs)
        rows.append((mode, r))
        print(f"{mode:10s} excl-PAD {r['loss_excl_pad']:.4f} / ppl "
              f"{r['ppl_excl_pad']:.3f}   incl-PAD {r['loss_incl_pad']:.4f} / ppl "
              f"{r['ppl_incl_pad']:.3f}   pad {100*r['pad_frac']:.1f}%", flush=True)

    pub = st.get("best_val_loss")
    md = [f"# Re-evaluation of published checkpoint — {label}", "",
          f"- checkpoint: `{args.ckpt}` (iter {st.get('iter_num')})",
          f"- val: `{args.val_lmdb}` ({rows[0][1]['n_seqs']:,} seqs)",
          f"- best_val_loss recorded in the checkpoint: {float(pub):.4f}" if pub is not None else "",
          "", "| batching | loss (excl PAD) | ppl (excl PAD) | loss (incl PAD) | ppl (incl PAD) | PAD fraction |",
          "|---|---:|---:|---:|---:|---:|"]
    for mode, r in rows:
        md.append(f"| {mode} | {r['loss_excl_pad']:.4f} | {r['ppl_excl_pad']:.3f} | "
                  f"{r['loss_incl_pad']:.4f} | {r['ppl_incl_pad']:.3f} | "
                  f"{100*r['pad_frac']:.1f}% |")
    ratio = rows[0][1]["ppl_excl_pad"] / max(rows[0][1]["ppl_incl_pad"], 1e-9)
    md += ["", f"**PAD dilution factor (pad-to-{cfg.block_size} convention): "
               f"{ratio:.1f}×** -- the PAD-dilution correction as a measured "
               "value rather than an estimate."]
    text = "\n".join(x for x in md if x != "")
    if args.report:
        with open(args.report, "w") as fh:
            fh.write(text + "\n")
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
