#!/usr/bin/env python3
"""
04 - Write the codon text files train_codon.txt.gz / val_codon.txt.gz according
to the cluster split from 03.

The output format is identical to the original pipeline's archea_rna_seq.txt
(T->U, a space every 3 nt, one sequence per line); the only addition is gzip
compression.

val is shuffled as a whole with a fixed seed before being written: the val
DataLoader in train.py uses shuffle=False, and eval draws contiguous windows of
the form SubsetRandomSampler(range(i, i+batch_size)) (train.py:240-242).  If val
kept cluster order, a single window would consist entirely of homologous
sequences and the eval result would not be representative.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time

import numpy as np

TRANS = bytes.maketrans(b"T", b"U")


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def open_read(path):
    """Decompress in parallel with unpigz, an order of magnitude faster than
    python gzip."""
    p = subprocess.Popen(["unpigz", "-c", path], stdout=subprocess.PIPE, bufsize=1 << 22)
    return p, p.stdout


def open_write(path, threads):
    p = subprocess.Popen(["pigz", "-p", str(threads), "-6", "-c"],
                         stdin=subprocess.PIPE, stdout=open(path, "wb"), bufsize=1 << 22)
    return p, p.stdin


def load_val_hashes(split_path):
    import gzip
    val, n_train, n_val = [], 0, 0
    with gzip.open(split_path, "rt") as fh:
        fh.readline()
        for line in fh:
            sid, _rep, sp = line.rstrip("\n").split("\t")
            if sp == "val":
                val.append(h64(sid))
                n_val += 1
            else:
                n_train += 1
    hs = set(val)
    assert len(hs) == len(val), "64-bit hash collision among val seq_ids"
    return hs, n_train, n_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cds", required=True, help="cds.fasta.gz produced by 01")
    ap.add_argument("--split", required=True, help="split.tsv.gz produced by 03")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tmp-dir", required=True, help="temporary directory for the val shuffle (node-local disk)")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    val_h, n_train_exp, n_val_exp = load_val_hashes(args.split)
    print(f"[{args.domain}] split: train={n_train_exp} val={n_val_exp} "
          f"({time.time()-t0:.0f}s)", flush=True)

    train_path = os.path.join(args.out_dir, "train_codon.txt.gz")
    val_path = os.path.join(args.out_dir, "val_codon.txt.gz")
    val_tmp = os.path.join(args.tmp_dir, "val_unshuffled.txt")

    rp, rfh = open_read(args.cds)
    tp, tfh = open_write(train_path, args.threads)
    vfh = open(val_tmp, "wb", buffering=1 << 22)

    val_offsets = []
    n_train = n_val = 0
    pos = 0
    sid = None
    chunks = []

    def emit():
        nonlocal n_train, n_val, pos
        if sid is None:
            return
        seq = b"".join(chunks).translate(TRANS)
        line = b" ".join(seq[i:i + 3] for i in range(0, len(seq), 3)) + b"\n"
        if h64(sid) in val_h:
            val_offsets.append((pos, len(line)))
            vfh.write(line)
            pos += len(line)
            n_val += 1
        else:
            tfh.write(line)
            n_train += 1

    for line in rfh:
        if line[:1] == b">":
            emit()
            sid = line[1:].rstrip().decode()
            chunks = []
        else:
            chunks.append(line.strip())
    emit()

    rfh.close(); rp.wait()
    tfh.close(); tp.wait()
    vfh.close()
    print(f"[{args.domain}] streamed: train={n_train} val={n_val} "
          f"({time.time()-t0:.0f}s)", flush=True)
    assert n_train == n_train_exp and n_val == n_val_exp, \
        f"count mismatch: train {n_train}!={n_train_exp}, val {n_val}!={n_val_exp}"

    # ---- write val out, shuffled with a fixed seed ----
    offs = np.asarray([o for o, _ in val_offsets], dtype=np.int64)
    lens = np.asarray([l for _, l in val_offsets], dtype=np.int32)
    perm = np.random.default_rng(args.seed).permutation(len(offs))
    vp, vout = open_write(val_path, args.threads)
    with open(val_tmp, "rb") as src:
        for k, i in enumerate(perm):
            src.seek(int(offs[i]))
            vout.write(src.read(int(lens[i])))
            if (k + 1) % 2_000_000 == 0:
                print(f"  val shuffled {k+1}/{len(perm)} ({time.time()-t0:.0f}s)", flush=True)
    vout.close(); vp.wait()
    os.remove(val_tmp)

    with open(args.report, "w") as r:
        r.write(f"# 04_write_codon_txt - {args.domain}\n\n")
        r.write(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        r.write(f"- train: {n_train:,} seqs -> `{train_path}` "
                f"({os.path.getsize(train_path)/2**30:.2f} GiB)\n")
        r.write(f"- val:   {n_val:,} seqs -> `{val_path}` "
                f"({os.path.getsize(val_path)/2**30:.2f} GiB), "
                f"shuffled as a whole with seed={args.seed}\n")
        r.write(f"- format: T->U, a space every 3 nt, one sequence per line (as in the original archea_rna_seq.txt)\n")
    print(f"[{args.domain}] DONE ({time.time()-t0:.0f}s)", flush=True)
    print(open(args.report).read())


if __name__ == "__main__":
    main()
