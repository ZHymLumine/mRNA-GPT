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
        # Diagnostic only.  A model trained with ignore_index=PAD never learns
        # to emit PAD, so it scores PAD positions badly and `loss_incl_pad`
        # comes out *above* `loss`.  `loss` is the number to report.
        "loss_incl_pad": loss_all,
        "ppl_incl_pad": float(torch.exp(torch.tensor(loss_all))),
        "loss_pad_positions": nll_pad / max(n_pad, 1.0),
        "tokens": int(n_real),
        "pad_tokens": int(n_pad),
        "seqs": int(float(n_seq)),
    }
