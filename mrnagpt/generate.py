"""Sampling: unconstrained, and constrained to a target protein.

Constrained decoding produces a CDS optimised for a *given* amino-acid sequence
rather than de novo generation of an arbitrary protein.  Because tokenization is
codon-level, one token is exactly one residue position, so this needs no change
to the architecture or to pretraining: at step t the logits are masked to the
synonymous codon set of residue t.  The translated output equals the target **by
construction**, not with high probability.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrnagpt.model import GPT, GPTConfig, KVCache            # noqa: E402
from mrnagpt.vocab import (BOS_ID, EOS_ID, GENETIC_CODE, PAD_ID, STOP_CODONS,  # noqa: E402
                           SYM2I, SYMBOLS, TOK2ID, UNK_ID, VOCAB_SIZE,
                           codons_for_symbol, decode_codons, translate)


# --------------------------------------------------------------------------- #
def build_allowed_mask(device) -> torch.Tensor:
    """(n_symbols, vocab_size) bool: which codon ids each residue symbol permits."""
    m = torch.zeros(len(SYMBOLS), VOCAB_SIZE, dtype=torch.bool, device=device)
    for sym, i in SYM2I.items():
        for c in codons_for_symbol(sym):
            m[i, TOK2ID[c]] = True
    assert m.any(dim=1).all(), "a residue symbol has an empty codon set"
    return m


def top_k_top_p_filter(logits: torch.Tensor, top_k: int | None,
                       top_p: float | None) -> torch.Tensor:
    """Must run *after* the constraint mask.  Applied before it, top-k could keep
    only disallowed codons and leave an all -inf row, which makes multinomial NaN."""
    if top_k:
        n_allowed = torch.isfinite(logits).sum(dim=-1)
        k = int(min(top_k, int(n_allowed.max())))
        if k > 0:
            kth = logits.topk(k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and 0.0 < top_p < 1.0:
        srt, idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(srt, dim=-1).cumsum(dim=-1)
        drop = probs - F.softmax(srt, dim=-1) >= top_p
        drop[..., 0] = False                       # always keep the top token
        logits = logits.scatter(-1, idx, srt.masked_fill(drop, float("-inf")))
    return logits


def _forward_step(model, idx, cache, window_mode, n_anchor, block_size):
    if cache is not None:
        feed = idx[:, cache.length:]
        logits, _ = model(feed, cache=cache, pos_offset=cache.length)
        return logits[:, -1, :]
    if idx.size(1) <= block_size:
        cond = idx
    elif window_mode == "rope_extend":
        cond = idx
    elif window_mode == "truncate":
        cond = idx[:, -block_size:]
    else:                                          # "anchor"
        # keep BOS at absolute position 0; a naive truncation would present a
        # window starting mid-CDS as if it were the start of a transcript
        cond = torch.cat([idx[:, :n_anchor], idx[:, -(block_size - n_anchor):]], dim=1)
    logits, _ = model(cond)
    return logits[:, -1, :]


@torch.no_grad()
def constrained_sample(model, proteins: list[str], *, temperature: float = 1.0,
                       top_k: int | None = None, top_p: float | None = None,
                       force_atg: bool = True, add_stop: bool = True,
                       window_mode: str = "anchor", n_anchor: int = 1,
                       use_cache: bool = True, device: str = "cuda",
                       generator: torch.Generator | None = None,
                       batch_size: int = 32) -> list[list[str]]:
    """Return, per input protein, the codon list whose translation is that protein.

    Chunked: the KV cache is (B, n_head, L, head_dim) per layer, so a whole batch
    of long sequences at once is tens to hundreds of GB.  Sequences are also
    grouped by length, since a chunk costs its longest member.
    """
    if len(proteins) > batch_size:
        order = sorted(range(len(proteins)), key=lambda i: len(proteins[i]))
        out: list[list[str] | None] = [None] * len(proteins)
        for s0 in range(0, len(order), batch_size):
            idx = order[s0:s0 + batch_size]
            got = constrained_sample(
                model, [proteins[i] for i in idx], temperature=temperature,
                top_k=top_k, top_p=top_p, force_atg=force_atg, add_stop=add_stop,
                window_mode=window_mode, n_anchor=n_anchor, use_cache=use_cache,
                device=device, generator=generator, batch_size=batch_size)
            for i, g in zip(idx, got):
                out[i] = g
        return out  # type: ignore[return-value]

    cfg = model.config if hasattr(model, "config") else model.module.config
    allowed_all = build_allowed_mask(device)
    B = len(proteins)
    Lmax = max(len(p) for p in proteins)
    n_steps = Lmax + (1 if add_stop else 0)

    # 'X' (any sense codon) fills the tail of shorter rows; those tokens are sliced off
    sym = torch.full((B, n_steps), SYM2I["X"], dtype=torch.long, device=device)
    for b, p in enumerate(proteins):
        if p:
            sym[b, :len(p)] = torch.tensor([SYM2I[a] for a in p], device=device)
        if add_stop:
            sym[b, len(p)] = SYM2I["*"]

    idx = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
    total_len = n_steps + 2
    cache = None
    if use_cache and total_len <= cfg.block_size:
        p0 = next(model.parameters())
        cache = KVCache(cfg, B, total_len, device, p0.dtype)

    only_atg = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    only_atg[TOK2ID["AUG"]] = True
    first_is_met = torch.tensor([p[:1] == "M" for p in proteins], device=device)

    for t in range(n_steps):
        logits = _forward_step(model, idx, cache, window_mode, n_anchor,
                               cfg.block_size).float()
        logits = logits / max(temperature, 1e-6)
        allowed = allowed_all[sym[:, t]]
        if force_atg and t == 0:
            allowed = torch.where(first_is_met[:, None], allowed & only_atg, allowed)
        logits = logits.masked_fill(~allowed, float("-inf"))
        if not torch.isfinite(logits).any(dim=-1).all():
            raise RuntimeError(f"empty allowed set at position {t}")
        logits = top_k_top_p_filter(logits, top_k, top_p)
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=generator)
        idx = torch.cat([idx, nxt], dim=1)

    out = []
    for b, p in enumerate(proteins):
        n = len(p) + (1 if add_stop else 0)
        out.append(decode_codons(idx[b, 1:1 + n]))
    return out


@torch.no_grad()
def prefix_sample(model, prefixes: list[list[str]], *, n_codons: int = 300,
                  temperature: float = 1.0, top_k: int | None = None,
                  top_p: float | None = None, device: str = "cuda",
                  generator: torch.Generator | None = None,
                  use_cache: bool = True, batch_size: int = 32
                  ) -> list[list[str]]:
    """Continue each codon prefix for ``n_codons`` codons.

    The pretrained models carry no species token -- nothing in the vocabulary
    names an organism -- so the only way to ask one of them for *this species'*
    sequences is to condition on text from that species.  Each prefix is the
    opening codons of one real CDS; what the model writes after it is the
    species-conditioned generation, and only that part is returned.

    EOS is banned: a short continuation would give a codon-usage vector made of
    noise, and the point here is the codon usage of a fixed-length sample.
    """
    if len(prefixes) > batch_size:
        out: list[list[str]] = []
        for s0 in range(0, len(prefixes), batch_size):
            out.extend(prefix_sample(
                model, prefixes[s0:s0 + batch_size], n_codons=n_codons,
                temperature=temperature, top_k=top_k, top_p=top_p, device=device,
                generator=generator, use_cache=use_cache, batch_size=batch_size))
        return out

    cfg = model.config if hasattr(model, "config") else model.module.config
    B = len(prefixes)
    Lp = len(prefixes[0])
    assert all(len(p) == Lp for p in prefixes), "prefixes must share a length"
    idx = torch.tensor([[BOS_ID] + [TOK2ID.get(c, UNK_ID) for c in p]
                        for p in prefixes], dtype=torch.long, device=device)
    total = min(1 + Lp + n_codons, cfg.block_size)
    cache = None
    if use_cache:
        p0 = next(model.parameters())
        cache = KVCache(cfg, B, total, device, p0.dtype)
    banned = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    banned[[PAD_ID, UNK_ID, BOS_ID, EOS_ID]] = True

    for _ in range(total - idx.size(1)):
        logits = _forward_step(model, idx, cache, "anchor", 1, cfg.block_size).float()
        logits = (logits / max(temperature, 1e-6)).masked_fill(banned, float("-inf"))
        logits = top_k_top_p_filter(logits, top_k, top_p)
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
        idx = torch.cat([idx, nxt], dim=1)
    return [decode_codons(row[1 + Lp:]) for row in idx]


@torch.no_grad()
def sample(model, n: int = 8, max_codons: int = 2044, temperature: float = 1.0,
           top_k: int | None = None, top_p: float | None = None,
           device: str = "cuda", generator: torch.Generator | None = None,
           use_cache: bool = True, batch_size: int = 32) -> list[list[str]]:
    """Unconstrained de novo generation, terminating on [EOS].

    Chunked for the same reason as constrained_sample: at n=1000 and
    max_codons=2044 a single KV cache is (1000, 16, 2046, 64) x 24 layers x 2,
    which is ~400 GB.
    """
    if n > batch_size:
        out: list[list[str]] = []
        for s0 in range(0, n, batch_size):
            out.extend(sample(model, n=min(batch_size, n - s0), max_codons=max_codons,
                              temperature=temperature, top_k=top_k, top_p=top_p,
                              device=device, generator=generator,
                              use_cache=use_cache, batch_size=batch_size))
        return out

    cfg = model.config if hasattr(model, "config") else model.module.config
    idx = torch.full((n, 1), BOS_ID, dtype=torch.long, device=device)
    total = min(max_codons, cfg.block_size - 2) + 2
    cache = None
    if use_cache:
        p0 = next(model.parameters())
        cache = KVCache(cfg, n, total, device, p0.dtype)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    # never emit PAD, UNK or BOS mid-sequence
    banned = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    banned[[PAD_ID, UNK_ID, BOS_ID]] = True

    for _ in range(total - 2):
        logits = _forward_step(model, idx, cache, "anchor", 1, cfg.block_size).float()
        logits = (logits / max(temperature, 1e-6)).masked_fill(banned, float("-inf"))
        logits = top_k_top_p_filter(logits, top_k, top_p)
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
        nxt[done] = EOS_ID
        idx = torch.cat([idx, nxt], dim=1)
        done |= nxt.squeeze(1) == EOS_ID
        if bool(done.all()):
            break
    return [decode_codons(row[1:][:(row[1:] == EOS_ID).long().argmax()
                                  if bool((row[1:] == EOS_ID).any()) else len(row) - 1])
            for row in idx]


# --------------------------------------------------------------------------- #
def gc_stats(codons) -> tuple[float, float]:
    seq = "".join(codons)
    if not seq:
        return 0.0, 0.0
    gc = sum(c in "GC" for c in seq) / len(seq)
    third = [c[2] for c in codons if len(c) == 3]
    gc3 = sum(c in "GC" for c in third) / max(len(third), 1)
    return gc, gc3


def validate_cds(codons, target_protein: str | None = None) -> dict:
    """Syntactic validity of a generated CDS.

    Columns mirror ``scripts/08_qc_stats.py`` so generated sequences line up with
    the training-set baselines in ``reports/*_qc_stats.md``.
    """
    codons = list(codons)
    starts_atg = codons[:1] == ["AUG"]
    ends_stop = bool(codons) and codons[-1] in STOP_CODONS
    internal_stop = any(c in STOP_CODONS for c in codons[:-1])
    gc, gc3 = gc_stats(codons)
    protein = translate(codons)
    match = None
    if target_protein is not None:
        body = codons[:-1] if ends_stop else codons
        match = translate(body) == target_protein
    return {
        "n_codon": len(codons),
        "starts_atg": starts_atg,
        "ends_stop": ends_stop,
        "internal_stop": internal_stop,
        "has_unknown": any(c not in GENETIC_CODE for c in codons),
        "valid_cds": starts_atg and ends_stop and not internal_stop,
        "protein": protein,
        "protein_match": match,
        "gc_content": gc,
        "gc3": gc3,
    }


def validate_batch(records, label: str = "generated") -> tuple[list[dict], str]:
    rows = [validate_cds(c, t) for c, t in records]
    n = max(len(rows), 1)

    def pct(key):
        return 100.0 * sum(bool(r[key]) for r in rows) / n

    md = [f"## CDS syntactic validity — {label}", "",
          "| | starts with ATG | ends with a stop codon "
          "| has an internal stop codon | **all three satisfied** |",
          "|---|---:|---:|---:|---:|",
          f"| {label} | {pct('starts_atg'):.2f}% | {pct('ends_stop'):.2f}% | "
          f"{pct('internal_stop'):.2f}% | **{pct('valid_cds'):.2f}%** |", ""]
    matches = [r["protein_match"] for r in rows if r["protein_match"] is not None]
    if matches:
        md.append(f"- target protein exact match: "
                  f"{100.0*sum(matches)/len(matches):.2f}% "
                  f"({sum(matches)}/{len(matches)})")
    lens = [r["n_codon"] for r in rows]
    gcs = [r["gc_content"] for r in rows]
    md.append(f"- sequences {n}, codon length mean {sum(lens)/n:.1f}, "
              f"GC {100*sum(gcs)/n:.2f}%, GC3 {100*sum(r['gc3'] for r in rows)/n:.2f}%")
    return rows, "\n".join(md)


def read_proteins(path: str) -> tuple[list[str], list[str]]:
    """Read target proteins from FASTA, or from one sequence per line.

    FASTA is what sequences usually arrive as, so it is detected rather than
    requiring the caller to strip headers first; a file whose first
    non-blank character is ``>`` is parsed as FASTA, anything else as one
    sequence per line. Returns the sequences and a name for each, the latter
    used to label the generated records.
    """
    with open(path) as fh:
        lines = [l.strip() for l in fh if l.strip()]
    if not lines:
        raise ValueError(f"{path} contains no sequences")

    if lines[0].startswith(">"):
        names, seqs, cur = [], [], []
        for line in lines:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
                names.append(line[1:].split()[0] or f"seq{len(names) + 1}")
            else:
                cur.append(line.upper())
        if cur:
            seqs.append("".join(cur))
        if len(seqs) != len(names):
            raise ValueError(f"{path}: a FASTA header has no sequence under it")
    else:
        seqs = [l.upper() for l in lines]
        names = [f"seq{i + 1}" for i in range(len(seqs))]

    unknown = {c for s in seqs for c in s} - set(SYMBOLS)
    if unknown:
        raise ValueError(f"{path}: unknown residue symbol(s) "
                         f"{''.join(sorted(unknown))}; expected amino acids, "
                         f"not nucleotides")
    return seqs, names


def _load_safetensors(directory: str, device: str) -> tuple[dict, dict]:
    """Published layout: ``config.json`` beside ``model.safetensors``.

    The export drops the tied ``lm_head.weight``, because safetensors refuses to
    serialise two names backed by the same storage. Tying is recorded in
    config.json and re-established when the model is constructed, so the key is
    restored by construction rather than by the state dict.
    """
    import json
    from safetensors.torch import load_file

    with open(os.path.join(directory, "config.json")) as fh:
        raw = json.load(fh)
    known = {f.name for f in dataclasses.fields(GPTConfig)}
    args = {k: v for k, v in raw.items() if k in known}
    return args, load_file(os.path.join(directory, "model.safetensors"),
                           device=device)


def load_model(ckpt_path: str, device: str = "cuda") -> GPT:
    """Load from a published model directory or a training checkpoint.

    ``ckpt_path`` may be a directory holding ``config.json`` and
    ``model.safetensors`` (what ``huggingface-cli download`` produces), the
    ``model.safetensors`` file inside such a directory, or a ``.pt`` checkpoint
    written during training.
    """
    directory = None
    if os.path.isdir(ckpt_path):
        directory = ckpt_path
    elif ckpt_path.endswith(".safetensors"):
        directory = os.path.dirname(os.path.abspath(ckpt_path))

    if directory is not None:
        model_args, sd = _load_safetensors(directory, device)
    else:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_args = state["model_args"]
        sd = {(k[10:] if k.startswith("_orig_mod.") else k): v
              for k, v in state["model"].items()}

    model = GPT(GPTConfig(**model_args)).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in {ckpt_path}: {unexpected[:5]}")
    # lm_head.weight is tied to transformer.wte.weight, so it is populated by
    # loading the latter; anything else missing means the wrong file.
    unresolved = [k for k in missing if k != "lm_head.weight"]
    if unresolved:
        raise RuntimeError(f"missing keys in {ckpt_path}: {unresolved[:5]}")
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None, help="FASTA output path")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--proteins", default=None,
                    help="file of target amino-acid sequences, one per line")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--report", default=None)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="generation chunk size; the KV cache scales with it")
    ap.add_argument("--max-codons", type=int, default=2044)
    args = ap.parse_args()

    model = load_model(args.ckpt, args.device)
    if args.proteins:
        prots, names = read_proteins(args.proteins)
        outs = constrained_sample(model, prots, temperature=args.temperature,
                                  top_k=args.top_k, top_p=args.top_p,
                                  device=args.device, batch_size=args.batch_size)
        records = list(zip(outs, prots))
        label = "constrained"
        out_names = names
    else:
        outs = sample(model, n=args.n, temperature=args.temperature,
                      top_k=args.top_k, top_p=args.top_p, device=args.device,
                      batch_size=args.batch_size, max_codons=args.max_codons)
        records = [(o, None) for o in outs]
        label = "unconstrained"
        out_names = None

    rows, md = validate_batch(records, label)
    print(md)
    if args.report:
        open(args.report, "w").write(md + "\n")
    if args.out:
        with open(args.out, "w") as fh:
            for i, (codons, _) in enumerate(records):
                # a constrained run carries the target's name through, so the
                # design can be matched back to the protein it was made for
                name = out_names[i] if out_names else f"{label}_{i}"
                fh.write(f">{name} n_codon={len(codons)}\n{''.join(codons)}\n")
        print(f"\nwrote {len(records)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
