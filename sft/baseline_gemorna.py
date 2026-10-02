"""Baseline 3: GEMORNA (RainaBio), zero-shot CDS generation from a target protein.

GEMORNA is a protein->CDS seq2seq generator; its decoder is NOT constrained to
the synonymous codon set of the target residue, so amino-acid identity is a
property to MEASURE, not to assume (contrast mrnagpt/generate.py, where masking
makes it exact by construction). Every sequence is translated and compared to
the requested protein, and the per-target identity rate is reported.

The released CDS checkpoint is stochastic: repeated runs on the same protein
give different sequences, so n variants per target are produced the same way as
for the model, by running generation n times.

Generation is a python loop over a small transformer on CPU (~0.4 s/residue), so
tasks are spread over a process pool; each worker loads the checkpoint once.
Must run in a dedicated environment: the generation code ships as a cpython-310
.so and the vocabularies are torchtext 0.6 pickles.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import pickle
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sft.paths import external                                   # noqa: E402

# GEMORNA is third-party software and is not included in this repository:
# obtain it from RainaBio, then set GEMORNA_DIR to the checkout -- or put it at
# <MRNA_GPT_EXTERNAL>/GEMORNA.
GEMORNA_DIR = str(external("GEMORNA", "GEMORNA_DIR"))

_state: dict = {}


def _init(ckpt_path: str):
    import torch
    sys.path.insert(0, os.path.join(GEMORNA_DIR, "src"))
    os.chdir(GEMORNA_DIR)
    torch.set_num_threads(1)
    from config import GEMORNA_CDS_Config
    from models.gemorna_cds import CDS, Decoder, Encoder

    cfg = GEMORNA_CDS_Config()
    dev = torch.device("cpu")
    enc = Encoder(input_dim=cfg.input_dim, hid_dim=cfg.hidden_dim, n_layers=cfg.num_layers,
                  n_heads=cfg.num_heads, pf_dim=cfg.ff_dim, dropout=cfg.dropout,
                  cnn_kernel_size=cfg.cnn_kernel_size, cnn_padding=cfg.cnn_padding, device=dev)
    dec = Decoder(output_dim=cfg.output_dim, hid_dim=cfg.hidden_dim, n_layers=cfg.num_layers,
                  n_heads=cfg.num_heads, pf_dim=cfg.ff_dim, dropout=cfg.dropout, device=dev)
    model = CDS(enc, dec, cfg.prot_pad_idx, cfg.cds_pad_idx, dev)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    model.eval()
    _state.update(model=model, dev=dev,
                  prot_vocab=pickle.load(open(os.path.join(GEMORNA_DIR, "vocab/prot_vocab.pkl"), "rb")),
                  cds_vocab=pickle.load(open(os.path.join(GEMORNA_DIR, "vocab/cds_vocab.pkl"), "rb")))


def _generate(task):
    target, protein, i = task
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _state["model"].gen(protein, _state["prot_vocab"], _state["cds_vocab"], _state["dev"])
    lines = [l for l in buf.getvalue().strip().splitlines() if l.strip()]
    parts = lines[-1].split()
    seq = parts[0].upper()
    naturalness = float(parts[1]) if len(parts) > 1 else None
    return target, i, seq, naturalness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(REPO, "data/protein_sequences.csv"))
    ap.add_argument("--ckpt", default=os.path.join(GEMORNA_DIR, "checkpoints/gemorna_cds.pt"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--targets", default=None, help="comma-separated subset of target slugs")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    from mrnagpt.vocab import GENETIC_CODE, STOP_CODONS
    from sft.generate_target_panel import load_targets

    targets = load_targets(args.csv)
    if args.targets:
        want = set(args.targets.split(","))
        targets = [(t, p) for t, p in targets if t in want]
    tasks = [(t, p, i) for t, p in targets for i in range(args.n)]
    print(f"GEMORNA: {len(targets)} targets x {args.n} = {len(tasks)} generations "
          f"on {args.workers} workers", flush=True)

    with mp.Pool(args.workers, initializer=_init, initargs=(args.ckpt,)) as pool:
        got = pool.map(_generate, tasks, chunksize=1)

    prot_of = dict(targets)
    per_target: dict = {t: {"n": 0, "protein_match": 0, "length_match": 0,
                            "naturalness": []} for t, _ in targets}
    label = "gemorna"
    with open(args.out, "w") as fh:
        for target, i, seq, nat in sorted(got, key=lambda r: (r[0], r[1])):
            rna = seq.replace("T", "U")
            codons = [rna[j:j + 3] for j in range(0, len(rna) - len(rna) % 3, 3)]
            body = codons[:-1] if codons and codons[-1] in STOP_CODONS else codons
            translated = "".join(GENETIC_CODE.get(c, "?") for c in body)
            st = per_target[target]
            st["n"] += 1
            st["protein_match"] += int(translated == prot_of[target])
            st["length_match"] += int(len(translated) == len(prot_of[target]))
            if nat is not None:
                st["naturalness"].append(nat)
            # no terminal stop is emitted; append one so CDS validity is not failed
            # for a reason unrelated to the method (same treatment as LinearDesign)
            out_codons = codons + ([] if (codons and codons[-1] in STOP_CODONS) else ["UAA"])
            fh.write(f">{label}|{target}|{i} n_codon={len(out_codons)}\n{''.join(out_codons)}\n")

    summary = {}
    for t, st in per_target.items():
        n = max(st["n"], 1)
        summary[t] = {"n": st["n"],
                      "protein_match_pct": 100.0 * st["protein_match"] / n,
                      "length_match_pct": 100.0 * st["length_match"] / n,
                      "mean_naturalness": (sum(st["naturalness"]) / len(st["naturalness"])
                                           if st["naturalness"] else None)}
        print(f"{t}: n={st['n']} protein-identity {summary[t]['protein_match_pct']:.1f}% "
              f"length-match {summary[t]['length_match_pct']:.1f}% "
              f"naturalness {summary[t]['mean_naturalness']}", flush=True)
    with open(args.out + ".meta.json", "w") as fh:
        json.dump({"stop_codon_appended": "UAA", "per_target": summary}, fh, indent=2)
    print(f"wrote {len(got)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
