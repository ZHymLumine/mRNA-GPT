#!/usr/bin/env python3
"""
07 - End-to-end verification: open the LMDB with the same read logic RNADataset
     uses in train.py, decode the tokens back to codons, and compare them
     sequence by sequence against the codon text produced by 04.

It also reports the PAD fraction, which is what makes reporting loss/perplexity
excluding PAD tokens necessary.
"""
import argparse
import subprocess

import lmdb
import numpy as np


def load_vocab(path):
    with open(path) as fh:
        return [l.rstrip("\n") for l in fh if l.rstrip("\n") != ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--txt", required=True, help="the matching *_codon.txt.gz (same order)")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--dtype", default="uint8")
    ap.add_argument("--block-size", type=int, default=2048)
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()

    toks = load_vocab(args.vocab)
    dtype = np.dtype(args.dtype)

    env = lmdb.open(args.lmdb, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin() as txn:
        entries = txn.stat()["entries"]
        print(f"LMDB entries: {entries:,}")

        p = subprocess.Popen(["unpigz", "-c", args.txt], stdout=subprocess.PIPE)
        pad_frac = []
        n_checked = 0
        for i in range(min(args.n, entries)):
            raw = txn.get(str(i).encode())
            assert raw is not None, f"key {i} missing"
            ids = np.frombuffer(raw, dtype=dtype)
            assert ids.max() < len(toks), f"token id {ids.max()} is outside the vocabulary of {len(toks)}"

            line = p.stdout.readline().decode().strip()
            expect = line.split()
            got = [toks[j] for j in ids]
            # encoding: [CLS] [SEP] <codons> [SEP] [SEP]
            assert got[:2] == ["[CLS]", "[SEP]"], got[:2]
            assert got[-2:] == ["[SEP]", "[SEP]"], got[-2:]
            assert got[2:-2] == expect, f"codons of record {i} do not match"

            # reproduce the pad/truncate of RNADataset in train.py
            d = ids[:args.block_size] if len(ids) > args.block_size else \
                np.pad(ids, (0, args.block_size - len(ids)), constant_values=0)
            assert len(d) == args.block_size
            pad_frac.append(1 - len(ids) / args.block_size)
            n_checked += 1
        p.stdout.close(); p.kill()

    print(f"OK: {n_checked:,} LMDB records decode to exactly the codons in {args.txt}")
    print(f"    mean PAD fraction at block_size={args.block_size}: {np.mean(pad_frac):.1%}"
          f" (median {np.median(pad_frac):.1%})")


if __name__ == "__main__":
    main()
