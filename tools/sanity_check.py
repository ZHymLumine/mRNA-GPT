"""Sanity-check a trained checkpoint against two baselines:

  1. an untrained (freshly initialized) model of the same architecture, which
     should score close to ln(vocab_size) = uniform-guessing perplexity;
  2. an order-0 unigram baseline that always predicts the training-set codon
     frequency distribution, regardless of context -- any model that has
     actually learned sequential/positional structure must clearly beat this.

Also reports top-1 next-codon accuracy on the validation set, which is a more
intuitive number than perplexity for judging "did it learn anything."
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import lmdb
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrnagpt.data import CodonLMDBDataset, build_plan, make_loader  # noqa: E402
from mrnagpt.model import GPT, GPTConfig                             # noqa: E402
from mrnagpt.vocab import CODON0, PAD_ID, VOCAB_SIZE                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("MRNA_GPT_DATA") or os.path.join(ROOT, "data")
RUNS_ROOT = os.environ.get("MRNA_GPT_RUNS") or os.path.join(ROOT, "runs")


def load_model(ckpt_path, device):
    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**st["model_args"])
    m = GPT(cfg).to(device)
    sd = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in st["model"].items()}
    m.load_state_dict(sd)
    m.eval()
    return m, cfg


def codon_unigram_dist(train_lmdb_path, n_sample_seqs=200_000, seed=0):
    """Estimate the training-set codon frequency distribution from a random sample."""
    from mrnagpt.data import load_lengths
    lens = load_lengths(train_lmdb_path)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(lens), size=min(n_sample_seqs, len(lens)), replace=False)
    env = lmdb.open(train_lmdb_path, subdir=False, readonly=True, lock=False)
    counts = np.zeros(VOCAB_SIZE, dtype=np.int64)
    from mrnagpt.vocab import remap_entry
    with env.begin(buffers=True) as txn:
        for i in idx:
            raw = np.frombuffer(txn.get(b"%d" % int(i)), dtype=np.uint8)
            ids = remap_entry(raw)
            counts[ids[1:]] += 1   # predicted tokens only (exclude leading BOS)
    env.close()
    counts[PAD_ID] = 0
    probs = counts / counts.sum()
    return probs


@torch.no_grad()
def evaluate_all(model, unigram_log_probs, ds, device, token_budget, max_seqs, seed=999):
    # A fixed-seed shuffled plan, truncated to max_seqs: deterministic, and the
    # same prefix is reused for the untrained-model comparison call below, so all
    # three numbers (trained / untrained / unigram) are measured on one identical
    # subset.
    plan = build_plan(ds.lengths, token_budget, 1, 1, seed, 0)
    loader = make_loader(ds, plan, 0, num_workers=2, persistent=False)

    model_nll = model_n = 0.0
    uni_nll = uni_n = 0.0
    top1_correct = top1_n = 0
    seqs_done = 0
    for x, y, n_real in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logits, _ = model(x)
        tgt = y.reshape(-1)
        mask = tgt != PAD_ID
        per = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt, reduction="none")
        model_nll += float((per * mask).sum()); model_n += float(mask.sum())

        pred = logits.argmax(-1).reshape(-1)
        top1_correct += int(((pred == tgt) & mask).sum())
        top1_n += int(mask.sum())

        u = unigram_log_probs[tgt.clamp(min=0).cpu().numpy()]
        u = torch.from_numpy(u).to(device)
        uni_nll += float((-u * mask.float()).sum()); uni_n += float(mask.sum())

        seqs_done += x.size(0)
        if max_seqs and seqs_done >= max_seqs:
            break

    return {
        "model_loss": model_nll / model_n, "model_ppl": math.exp(model_nll / model_n),
        "unigram_loss": uni_nll / uni_n, "unigram_ppl": math.exp(uni_nll / uni_n),
        "top1_acc": top1_correct / top1_n, "n_tokens": int(model_n), "n_seqs": seqs_done,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max-seqs", type=int, default=20000)
    ap.add_argument("--token-budget", type=int, default=32768)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data_dir = args.data_dir or os.path.join(DATA_ROOT, args.domain)
    ckpt = args.ckpt or os.path.join(RUNS_ROOT, args.domain, "model_best.pt")

    print(f"[{args.domain}] loading trained checkpoint: {ckpt}")
    model, cfg = load_model(ckpt, args.device)

    print(f"[{args.domain}] estimating order-0 unigram baseline from training set")
    probs = codon_unigram_dist(os.path.join(data_dir, "train_codon.lmdb"))
    log_probs = np.log(np.clip(probs, 1e-12, None)).astype(np.float32)

    ds = CodonLMDBDataset(os.path.join(data_dir, "val_codon.lmdb"))
    print(f"[{args.domain}] evaluating trained model + unigram baseline on "
          f"{min(args.max_seqs, len(ds)):,} validation sequences")
    trained = evaluate_all(model, log_probs, ds, args.device, args.token_budget, args.max_seqs)

    print(f"[{args.domain}] building an UNTRAINED model of the same architecture for comparison")
    untrained = GPT(cfg).to(args.device).eval()
    fresh = evaluate_all(untrained, log_probs, ds, args.device, args.token_budget,
                         min(args.max_seqs, 5000))

    ln_v = math.log(VOCAB_SIZE - 1)  # -1: PAD never a target
    print(f"\n===== {args.domain} =====")
    print(f"{'':22s}{'loss':>10s}{'ppl':>10s}{'top-1 acc':>12s}")
    print(f"{'uniform-random ref':22s}{ln_v:10.4f}{math.exp(ln_v):10.2f}{'~'+format(1/(VOCAB_SIZE-1)*100,'.2f')+'%':>12s}")
    print(f"{'untrained model':22s}{fresh['model_loss']:10.4f}{fresh['model_ppl']:10.2f}{fresh['top1_acc']*100:11.2f}%")
    print(f"{'unigram baseline':22s}{trained['unigram_loss']:10.4f}{trained['unigram_ppl']:10.2f}{'':>12s}")
    print(f"{'TRAINED mRNA-GPT':22s}{trained['model_loss']:10.4f}{trained['model_ppl']:10.2f}{trained['top1_acc']*100:11.2f}%")
    print(f"\nevaluated on {trained['n_seqs']:,} val sequences / {trained['n_tokens']:,} real tokens")


if __name__ == "__main__":
    main()
