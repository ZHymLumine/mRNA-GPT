"""Baseline: CodonGPT (Nanil Therapeutics, NAR 2025), protein-constrained generation.

CodonGPT masks logits to the synonymous codon set of the target residue at each
step -- the same mechanism used in mrnagpt/generate.py -- so it is the most
directly comparable generative baseline: two models, same constraint, same
guarantee of amino-acid identity, different learned codon preferences.

The reference implementation shipped with the checkpoint
(synonymous_logit_processor.generate_candidate_codons_with_generate) calls
model.generate() once per codon, re-encoding the whole prefix each time; that is
O(L^2) forward passes and is impractical for a 676-residue target sampled 200
times. This module runs the identical decision rule in one incremental pass with
a KV cache. Equivalence is not assumed: --verify-equivalence reproduces the
reference implementation greedily and asserts the two give identical sequences.

Only the amino acid of each starting codon is used by the reference code, so
generation is conditioned on the protein alone -- no starting CDS is needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sft.paths import external                                   # noqa: E402

# CodonGPT (code and checkpoint) is third-party software and is not included in
# this repository: obtain it from its authors, then set CODONGPT_DIR to the
# checkout -- or put it at <MRNA_GPT_EXTERNAL>/codonGPT.
MODEL_DIR = str(external("codonGPT", "CODONGPT_DIR"))
sys.path.insert(0, MODEL_DIR)

STOPS = {"TAA", "TAG", "TGA"}


def load_model(device: str):
    import torch
    from transformers import GPT2LMHeadModel
    from tokenizer import CodonTokenizer
    from synonymous_logit_processor import aa_to_codon_human

    tok = CodonTokenizer()
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    model.eval().to(device)
    return model, tok, aa_to_codon_human


def _allowed_ids(tok, aa_to_codon, aa: str) -> list[int]:
    return tok.convert_tokens_to_ids(aa_to_codon[aa])


def generate_cached(model, tok, aa_to_codon, protein: str, *, device: str,
                    temperature: float = 1.0, top_k=None, top_p=None,
                    greedy: bool = False, generator=None, add_stop: bool = True) -> list[str]:
    """One incremental pass with a KV cache; same per-position rule as the reference."""
    import torch
    import torch.nn.functional as F

    ids = torch.tensor([[tok.bos_token_id]], device=device)
    past = None
    out_codons: list[str] = []
    symbols = list(protein) + (["*"] if add_stop else [])
    for aa in symbols:
        with torch.no_grad():
            res = model(input_ids=ids if past is None else ids[:, -1:], past_key_values=past,
                        use_cache=True)
        past = res.past_key_values
        logits = res.logits[:, -1, :].float()
        mask = torch.full_like(logits, float("-inf"))
        allowed = _allowed_ids(tok, aa_to_codon, aa)
        mask[:, allowed] = 0.0
        logits = logits + mask
        if greedy:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-6)
            if top_k:
                k = min(top_k, len(allowed))
                kth = logits.topk(k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p and 0.0 < top_p < 1.0:
                srt, idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(srt, dim=-1).cumsum(dim=-1)
                drop = probs - F.softmax(srt, dim=-1) >= top_p
                drop[..., 0] = False
                logits = logits.scatter(-1, idx, srt.masked_fill(drop, float("-inf")))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
        out_codons.append(tok.convert_ids_to_tokens(int(nxt))[0] if isinstance(
            tok.convert_ids_to_tokens(int(nxt)), list) else tok.convert_ids_to_tokens(int(nxt)))
        ids = torch.cat([ids, nxt], dim=1)
    return [c.upper() for c in out_codons]


def generate_reference(model, tok, aa_to_codon, protein: str, device: str,
                       add_stop: bool = True) -> list[str]:
    """The shipped per-codon model.generate() loop, greedy, for the equivalence check."""
    import torch
    from synonymous_logit_processor import SynonymMaskingLogitsProcessor

    seq_tokens = [tok.bos_token_id]
    out = []
    for aa in list(protein) + (["*"] if add_stop else []):
        lp = [SynonymMaskingLogitsProcessor(aa, tok, aa_to_codon)]
        input_ids = torch.tensor([seq_tokens], device=device)
        gen = model.generate(input_ids, max_length=len(seq_tokens) + 1,
                             num_return_sequences=1, pad_token_id=tok.pad_token_id,
                             logits_processor=lp, do_sample=False)
        nxt = int(gen[0][-1])
        out.append(tok.decode([nxt]).upper())
        seq_tokens.append(nxt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/protein_sequences.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify-equivalence", action="store_true",
                    help="greedy-decode one short target both ways and require identity")
    args = ap.parse_args()

    import torch
    from mrnagpt.vocab import GENETIC_CODE
    from sft.generate_target_panel import load_targets

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, aa_to_codon = load_model(device)
    targets = load_targets(args.csv)

    if args.verify_equivalence:
        name, prot = min(targets, key=lambda x: len(x[1]))
        probe = prot[:40]
        a = generate_cached(model, tok, aa_to_codon, probe, device=device, greedy=True)
        b = generate_reference(model, tok, aa_to_codon, probe, device)
        same = a == b
        print(f"[equivalence] {name}[:40] greedy: cached == reference -> {same}")
        if not same:
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f"  first mismatch at codon {i}: cached {x} vs reference {y}")
                    break
            raise SystemExit("cached implementation does not reproduce the reference")
        if not args.out:
            return

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    label = "codongpt"
    per_target = {}
    with open(args.out, "w") as fh:
        for target, prot in targets:
            ok = 0
            for i in range(args.n):
                codons = generate_cached(model, tok, aa_to_codon, prot, device=device,
                                         temperature=args.temperature, top_k=args.top_k,
                                         top_p=args.top_p, generator=gen)
                body = codons[:-1] if codons and codons[-1] in STOPS else codons
                rna = [c.replace("T", "U") for c in body]
                translated = "".join(GENETIC_CODE.get(c, "?") for c in rna)
                ok += int(translated == prot)
                fh.write(f">{label}|{target}|{i} n_codon={len(codons)}\n"
                         f"{''.join(codons).replace('T', 'U')}\n")
            per_target[target] = {"n": args.n, "protein_match_pct": 100.0 * ok / args.n}
            print(f"{target}: n={args.n} protein-identity {100.0 * ok / args.n:.1f}%", flush=True)
    with open(args.out + ".meta.json", "w") as fh:
        json.dump({"model": MODEL_DIR, "temperature": args.temperature,
                   "top_k": args.top_k, "top_p": args.top_p, "seed": args.seed,
                   "per_target": per_target}, fh, indent=2)
    print(f"wrote {args.n * len(targets)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
