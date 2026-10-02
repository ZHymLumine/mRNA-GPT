"""Evaluation passes.

Replaces the published ``estimate_loss``, which built a fresh DataLoader inside
every one of its 100 eval iterations and sampled a *contiguous* window via
``SubsetRandomSampler`` -- neither a random sample nor a full-set evaluation, and
~17 s per call (about 7% of total runtime).

Two conventions are reported side by side:

* ``loss``           -- mean NLL over real (non-PAD) target tokens.  The honest one.
* ``loss_incl_pad``  -- mean over every target position including PAD, i.e. what
  the published Figures S1-S3 plot.  Kept only so the two can be compared.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import CodonLMDBDataset, build_plan, make_loader
from .vocab import PAD_ID


@torch.no_grad()
def evaluate(model, dataset: CodonLMDBDataset, *, ddp, device, token_budget: int,
             seed: int = 12345, max_seqs: int = 0, num_workers: int = 4,
             autocast_dtype=torch.bfloat16) -> dict:
    """Deterministic sharded pass; every sequence is scored exactly once.

    Aggregates ``sum(nll)`` and ``sum(n_tokens)`` across ranks and divides at the
    end -- not a mean of means, which with variable-length buckets is a different
    number.
    """
    was_training = model.training
    model.eval()

    lengths = dataset.lengths
    if max_seqs and max_seqs < lengths.shape[0]:
        # frozen subset: same sequences at every eval, chosen once by a fixed seed
        rng = __import__("numpy").random.default_rng(seed)
        keep = rng.choice(lengths.shape[0], size=max_seqs, replace=False)
        keep.sort()
        sub_lengths = lengths[keep]
    else:
        keep = None
        sub_lengths = lengths

    plan = build_plan(sub_lengths, token_budget, ddp.world_size, 1,
                      seed=seed, epoch=0)
    if keep is not None:
        # remap plan indices back onto the parent dataset
        mapped = plan.order.copy()
        m = mapped >= 0
        mapped[m] = keep[mapped[m]]
        plan.order = mapped

    loader = make_loader(dataset, plan, ddp.rank, num_workers=num_workers,
                         persistent=False)
    acc = torch.zeros(4, dtype=torch.float64, device=device)   # nll_real, n_real, nll_pad, n_pad
    n_seq = torch.zeros(1, dtype=torch.float64, device=device)

    for batch in loader:
        if batch is None:
            continue
        x, y, _ = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=device != "cpu"):
            logits, _ = model(x)
        tgt = y.reshape(-1)
        per_tok = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                  tgt, reduction="none")
        mask = tgt != PAD_ID
        acc[0] += (per_tok * mask).sum().double()
        acc[1] += mask.sum().double()
        acc[2] += (per_tok * (~mask)).sum().double()
        acc[3] += (~mask).sum().double()
        n_seq += float((x[:, 0] != PAD_ID).sum())

    ddp.all_reduce(acc)
    ddp.all_reduce(n_seq)
    nll_real, n_real, nll_pad, n_pad = (float(v) for v in acc)
    loss = nll_real / max(n_real, 1.0)
    loss_all = (nll_real + nll_pad) / max(n_real + n_pad, 1.0)

    if was_training:
        model.train()
    return {
        "loss": loss,
        "ppl": float(torch.exp(torch.tensor(loss))),
        # Diagnostic, NOT the published convention.  A model trained with
        # ignore_index=PAD never learns to emit PAD, so it scores PAD positions
        # badly and `loss_incl_pad` comes out *above* `loss`.  The published runs
        # trained on PAD and predicted it near-perfectly, diluting their loss
        # downward.  Use tools/eval_legacy_ckpt.py on the published checkpoints
        # for the PAD-excluded perplexity number.
        "loss_incl_pad": loss_all,
        "ppl_incl_pad": float(torch.exp(torch.tensor(loss_all))),
        "loss_pad_positions": nll_pad / max(n_pad, 1.0),
        "tokens": int(n_real),
        "pad_tokens": int(n_pad),
        "seqs": int(float(n_seq)),
    }


@torch.no_grad()
def evaluate_legacy(model, dataset: CodonLMDBDataset, *, ddp, device,
                    block_size: int, n_seqs: int = 4096, seed: int = 999,
                    autocast_dtype=torch.bfloat16) -> dict:
    """Reproduce the published convention: pad naively to block_size, mean over
    every position.  Reported for shape comparison only -- it is *not* numerically
    comparable to the published figures, which also used a 69-token vocab,
    block_size 1024, and a leaky random validation split."""
    import numpy as np
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n_seqs, len(dataset)), replace=False)

    total_nll = torch.zeros(1, dtype=torch.float64, device=device)
    total_n = torch.zeros(1, dtype=torch.float64, device=device)
    per_rank = np.array_split(idx, ddp.world_size)[ddp.rank]
    B = max(1, 65536 // block_size)
    for s in range(0, len(per_rank), B):
        chunk = per_rank[s:s + B]
        x = torch.zeros(len(chunk), block_size, dtype=torch.long)
        y = torch.zeros(len(chunk), block_size, dtype=torch.long)
        for i, j in enumerate(chunk):
            ids, _ = dataset[(int(j), block_size)]
            ids = ids[:block_size]
            n = ids.shape[0]
            x[i, :n - 1] = torch.from_numpy(ids[:-1].astype("int64"))
            y[i, :n - 1] = torch.from_numpy(ids[1:].astype("int64"))
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=device != "cpu"):
            logits, _ = model(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                              y.reshape(-1), reduction="sum")
        total_nll += nll.double()
        total_n += y.numel()

    ddp.all_reduce(total_nll)
    ddp.all_reduce(total_n)
    loss = float(total_nll) / max(float(total_n), 1.0)
    if was_training:
        model.train()
    return {"legacy_loss": loss, "legacy_ppl": float(torch.exp(torch.tensor(loss))),
            "legacy_block_size": block_size, "legacy_n_seqs": int(len(idx))}
