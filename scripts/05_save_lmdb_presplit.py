#!/usr/bin/env python3
"""
05 - Write the already-split codon text into LMDB (the format train.py reads).

Differences from the original save_lmdb.py:
  1. Takes the pre-split train / val files instead of doing its own chunk-shuffle
     + 0.9 cut (the random split at save_lmdb.py:141-149 is exactly the one that
     let homologues straddle train and val).
  2. block_size defaults to 2048 (was 1024).
  3. Builds the codon -> id mapping straight from vocab.txt instead of calling
     tokenizer.encode line by line, which is 1-2 orders of magnitude faster; a
     sample is asserted token-by-token against BertTokenizerFast before writing.
  4. dtype defaults to uint8 (the vocabulary has only 69 tokens).  int32 would
     make the LMDB 4x larger.  This requires train.py to read a configurable
     dtype instead of np.frombuffer(value, dtype=np.int32).

The encoding is identical to the original pipeline: [CLS] [SEP] <codons> [SEP] [SEP]
(the original code passed tokenizer.encode("[SEP]" + text + "[SEP]") and the
tokenizer then added [CLS]/[SEP] itself).
"""
import argparse
import os
import subprocess
import sys
import time

import lmdb
import numpy as np

MAP_SIZE = 1 << 40  # 1 TiB cap; LMDB allocates sparsely, so this does not occupy disk


def load_vocab(vocab_file):
    with open(vocab_file) as fh:
        toks = [l.rstrip("\n") for l in fh if l.rstrip("\n") != ""]
    return {t: i for i, t in enumerate(toks)}, toks


def build_codon_table(vocab):
    """codon(bytes) -> id, falling back to [UNK] for unknown codons."""
    return {k.encode(): v for k, v in vocab.items() if len(k) == 3 and not k.startswith("[")}


def encode_line(line, table, unk, cls_id, sep_id, add_cls, block_size):
    codons = line.split()
    ids = [sep_id] + [table.get(c, unk) for c in codons] + [sep_id, sep_id]
    if add_cls:
        ids.insert(0, cls_id)
    return ids if len(ids) <= block_size else None


def verify_equivalence(sample_lines, table, unk, cls_id, sep_id, add_cls, block_size,
                       vocab_file, n=10000):
    from transformers import BertTokenizerFast
    tk = BertTokenizerFast(vocab_file=vocab_file, do_lower_case=False)
    if len(tk) != 69:
        sys.exit(f"tokenizer loaded only {len(tk)} tokens (expected 69). "
                 f"transformers 5.x cannot read this vocab.txt correctly; "
                 f"please install transformers==4.46.3")
    checked = 0
    for line in sample_lines[:n]:
        text = line.decode()
        ref = tk.encode("[SEP]" + text + "[SEP]")
        if not add_cls:
            ref = ref[1:]
        mine = encode_line(line, table, unk, cls_id, sep_id, add_cls, block_size)
        if mine is None:
            continue
        if mine != ref:
            sys.exit(f"encoding disagrees with BertTokenizerFast!\n  mine={mine[:20]}\n  ref ={ref[:20]}")
        checked += 1
    print(f"  [verify] {checked} sampled sequences match BertTokenizerFast token by token", flush=True)


def write_lmdb(txt_path, lmdb_path, table, unk, cls_id, sep_id, add_cls, block_size,
               dtype, limit=None, label=""):
    if os.path.exists(lmdb_path):
        os.remove(lmdb_path)
    env = lmdb.open(lmdb_path, subdir=False, readonly=False, lock=False,
                    readahead=False, meminit=False, map_size=MAP_SIZE)
    proc = subprocess.Popen(["unpigz", "-c", txt_path], stdout=subprocess.PIPE,
                            bufsize=1 << 22)
    n = n_skipped = 0
    lens = []
    t0 = time.time()
    txn = env.begin(write=True)
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        ids = encode_line(line, table, unk, cls_id, sep_id, add_cls, block_size)
        if ids is None:
            n_skipped += 1
            continue
        txn.put(str(n).encode(), np.asarray(ids, dtype=dtype).tobytes())
        lens.append(len(ids))
        n += 1
        if n % 500_000 == 0:
            txn.commit()
            txn = env.begin(write=True)
            print(f"  [{label}] {n:,} written ({time.time()-t0:.0f}s)", flush=True)
        if limit and n >= limit:
            break
    txn.commit()
    proc.stdout.close()
    proc.wait()
    entries = env.stat()["entries"]
    env.close()
    lens = np.asarray(lens)
    return n, n_skipped, entries, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-txt", required=True)
    ap.add_argument("--val-txt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--block-size", type=int, default=2048)
    ap.add_argument("--dtype", default="uint8", choices=["uint8", "uint16", "int32"])
    ap.add_argument("--no-cls", action="store_true",
                    help="omit [CLS] (a decoder-only model does not need it). "
                         "It is added by default so the published results reproduce.")
    ap.add_argument("--val-small", type=int, default=200_000)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    t0 = time.time()
    vocab, toks = load_vocab(args.vocab)
    assert len(toks) == 69, f"vocab.txt should hold 69 tokens, found {len(toks)}"
    table = build_codon_table(vocab)
    assert len(table) == 64, f"expected 64 codons, found {len(table)}"
    unk, cls_id, sep_id = vocab["[UNK]"], vocab["[CLS]"], vocab["[SEP]"]
    add_cls = not args.no_cls
    dtype = np.dtype(args.dtype)
    assert len(toks) <= np.iinfo(dtype).max + 1, f"{args.dtype} cannot hold {len(toks)} tokens"

    # ---- equivalence check ----
    p = subprocess.Popen(["unpigz", "-c", args.val_txt], stdout=subprocess.PIPE)
    sample = [next(p.stdout).strip() for _ in range(10000)]
    p.stdout.close(); p.kill()
    verify_equivalence(sample, table, unk, cls_id, sep_id, add_cls,
                       args.block_size, args.vocab)

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for label, txt, out, limit in [
        ("train", args.train_txt, "train_codon.lmdb", None),
        ("val", args.val_txt, "val_codon.lmdb", None),
        ("val_small", args.val_txt, "val_small.lmdb", args.val_small),
    ]:
        path = os.path.join(args.out_dir, out)
        n, skipped, entries, lens = write_lmdb(
            txt, path, table, unk, cls_id, sep_id, add_cls,
            args.block_size, dtype, limit=limit, label=label)
        assert entries == n, f"{label}: LMDB entries {entries} != written {n}"
        results[label] = (n, skipped, path, lens)
        print(f"[{args.domain}] {label}: {n:,} entries -> {path} "
              f"({os.path.getsize(path)/2**30:.2f} GiB, {time.time()-t0:.0f}s)", flush=True)

    with open(args.report, "w") as r:
        r.write(f"# 05_save_lmdb_presplit - {args.domain}\n\n")
        r.write(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        r.write(f"- block_size: {args.block_size}, dtype: {args.dtype}, "
                f"[CLS] added: {add_cls}\n")
        r.write(f"- encoding: `{'[CLS] ' if add_cls else ''}[SEP] <codons> [SEP] [SEP]`"
                f" (identical to the original pipeline; asserted token-by-token on a 10,000-sequence sample)\n\n")
        r.write("| split | sequences | dropped (too long) | file size | token length P50/P95/max |\n")
        r.write("|---|---:|---:|---:|---|\n")
        for label in ("train", "val", "val_small"):
            n, skipped, path, lens = results[label]
            r.write(f"| {label} | {n:,} | {skipped:,} | "
                    f"{os.path.getsize(path)/2**30:.2f} GiB | "
                    f"{int(np.percentile(lens,50))} / {int(np.percentile(lens,95))} / "
                    f"{int(lens.max())} |\n")
        pad_frac = 1 - results["train"][3].mean() / args.block_size
        r.write(f"\n- mean PAD fraction of train: **{pad_frac:.1%}** "
                f"(block_size={args.block_size}) -- which is why loss/perplexity "
                f"must be reported excluding PAD tokens.\n")
    print(open(args.report).read())


if __name__ == "__main__":
    main()
