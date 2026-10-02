#!/usr/bin/env python3
"""
01 - Extract, filter and exactly deduplicate NCBI CDS fasta within a domain,
producing the cds.fasta.gz and meta.tsv.gz that mmseqs consumes.

Differences from the original pipeline (process_data_to_species.py):
  * Reads only *_cds_from_genomic.fna (the original glob '*.fna' also pulled in
    the 702 whole-genome _genomic.fna files for archaea).
  * Keeps provenance: seq_id / assembly accession / protein_id / locus_tag /
    species taxonomy.
  * Exact within-domain deduplication (an earlier run had 86 of 1,000 generated
    sequences identical to a training sequence).
  * Records QC flags (ATG start / stop codon / internal stop codon) but does not
    filter on them, so the proportion of syntactically valid CDS can be reported.

Sequences stay in the DNA alphabet (ACGT) at this stage; the T->U conversion
happens in step 04 when the codon text is written.
"""
import argparse
import gzip
import hashlib
import multiprocessing as mp
import os
import re
import shutil
import sys
import time

import numpy as np

ACGT_RE = re.compile(rb"^[ACGT]+$")
ACC_RE = re.compile(r"^(GC[AF]_\d+\.\d+)_cds_from_genomic\.fna$")
PROTID_RE = re.compile(r"\[protein_id=([^\]]+)\]")
LOCUS_RE = re.compile(r"\[locus_tag=([^\]]+)\]")
STOPS = (b"TAA", b"TAG", b"TGA")

META_COLS = [
    "seq_id", "accession", "protein_id", "locus_tag",
    "len_nt", "n_codon", "starts_atg", "ends_stop", "internal_stop",
]


def has_internal_stop(seq: bytes) -> int:
    """Whether a stop codon appears anywhere except in the final codon."""
    for i in range(0, len(seq) - 3, 3):
        if seq[i:i + 3] in STOPS:
            return 1
    return 0


def iter_fasta(path):
    """Yield (header:str, seq:bytes). Reads binary to avoid the unicode decoding
    overhead of 105M records."""
    with open(path, "rb") as fh:
        header = None
        chunks = []
        for line in fh:
            if line[:1] == b">":
                if header is not None:
                    yield header, b"".join(chunks)
                header = line[1:].rstrip().decode("utf-8", "replace")
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, b"".join(chunks)


def process_chunk(task):
    """One worker handles several fna files and writes the shard's .fa / .meta /
    .hash triplet."""
    shard_idx, files, tmp_dir, max_codons = task
    fa_path = os.path.join(tmp_dir, f"shard_{shard_idx:05d}.fa")
    meta_path = os.path.join(tmp_dir, f"shard_{shard_idx:05d}.meta")
    hash_path = os.path.join(tmp_dir, f"shard_{shard_idx:05d}.hash")

    n_total = n_kept = n_bad_char = n_bad_frame = n_too_long = 0
    hashes = []

    with open(fa_path, "wb") as fa_out, open(meta_path, "w") as meta_out:
        for path in files:
            fname = os.path.basename(path)
            m = ACC_RE.match(fname)
            accession = m.group(1) if m else fname
            for rec_idx, (header, seq) in enumerate(iter_fasta(path), start=1):
                n_total += 1
                seq = seq.upper()
                if not ACGT_RE.match(seq):
                    n_bad_char += 1
                    continue
                if len(seq) % 3:
                    n_bad_frame += 1
                    continue
                n_codon = len(seq) // 3
                if n_codon > max_codons or n_codon == 0:
                    n_too_long += 1
                    continue

                seq_id = f"{accession}|{rec_idx}"
                pm = PROTID_RE.search(header)
                lm = LOCUS_RE.search(header)
                fa_out.write(b">" + seq_id.encode() + b"\n" + seq + b"\n")
                meta_out.write("\t".join((
                    seq_id,
                    accession,
                    pm.group(1) if pm else "",
                    lm.group(1) if lm else "",
                    str(len(seq)),
                    str(n_codon),
                    "1" if seq[:3] == b"ATG" else "0",
                    "1" if seq[-3:] in STOPS else "0",
                    str(has_internal_stop(seq)),
                )) + "\n")
                hashes.append(int.from_bytes(
                    hashlib.blake2b(seq, digest_size=8).digest(), "little"))
                n_kept += 1

    np.asarray(hashes, dtype=np.uint64).tofile(hash_path)
    return shard_idx, n_total, n_kept, n_bad_char, n_bad_frame, n_too_long


_TAX = None


def _emit_init(tax):
    global _TAX
    _TAX = tax


def emit_shard(task):
    """Compress the records of one shard that keep_mask retains into their own
    .keep.fa.gz / .keep.meta.gz."""
    shard_idx, offset, n_rec, tmp_dir = task
    keep = np.load(os.path.join(tmp_dir, "keep_mask.npy"), mmap_mode="r")[offset:offset + n_rec]
    dup = np.load(os.path.join(tmp_dir, "dup_count.npy"), mmap_mode="r")[offset:offset + n_rec]

    fa_in = os.path.join(tmp_dir, f"shard_{shard_idx:05d}.fa")
    meta_in = os.path.join(tmp_dir, f"shard_{shard_idx:05d}.meta")
    fa_out_p = os.path.join(tmp_dir, f"shard_{shard_idx:05d}.keep.fa.gz")
    meta_out_p = os.path.join(tmp_dir, f"shard_{shard_idx:05d}.keep.meta.gz")

    written = 0
    unmatched = set()
    with gzip.open(fa_out_p, "wb", compresslevel=6) as fa_out, \
         gzip.open(meta_out_p, "wt", compresslevel=6) as meta_out, \
         open(meta_in) as mfh:
        for k, ((hdr, seq), mline) in enumerate(zip(iter_fasta(fa_in), mfh)):
            if not keep[k]:
                continue
            fa_out.write(b">" + hdr.encode() + b"\n" + seq + b"\n")
            parts = mline.rstrip("\n").split("\t")
            t = _TAX.get(parts[1])
            if t is None:
                unmatched.add(parts[1])
                t = ("", "", "", "")
            meta_out.write("\t".join(parts + [str(int(dup[k]))] + list(t)) + "\n")
            written += 1
    return shard_idx, written, unmatched


def load_taxonomy(csv_path):
    """Accession -> (taxid, superkingdom, phylum, species)"""
    import pandas as pd
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df = df.drop_duplicates(subset="Accession", keep="first")
    return {
        r["Accession"]: (r.get("Tax ID", ""), r.get("superkingdom", ""),
                         r.get("phylum", ""), r.get("species", ""))
        for _, r in df.iterrows()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="directory containing *_cds_from_genomic.fna")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tmp-dir", required=True, help="directory for intermediate shard files (node-local disk recommended)")
    ap.add_argument("--species-csv", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--max-codons", type=int, default=2044,
                    help="block_size 2048 - 4 special tokens")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--files-per-shard", type=int, default=0,
                    help="0 = automatic (number of files / (procs*4))")
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)

    files = sorted(
        os.path.join(args.raw_dir, f)
        for f in os.listdir(args.raw_dir)
        if f.endswith("_cds_from_genomic.fna")
    )
    if not files:
        sys.exit(f"no *_cds_from_genomic.fna under {args.raw_dir}")
    print(f"[{args.domain}] {len(files)} CDS fasta files", flush=True)

    fps = args.files_per_shard or max(1, len(files) // (args.procs * 4))
    tasks = [
        (i, files[s:s + fps], args.tmp_dir, args.max_codons)
        for i, s in enumerate(range(0, len(files), fps))
    ]
    print(f"[{args.domain}] {len(tasks)} shards x ~{fps} files, {args.procs} procs", flush=True)

    # ---- pass 1: parse and filter in parallel, write shards ----
    stats = {}
    with mp.Pool(args.procs) as pool:
        for n, res in enumerate(pool.imap_unordered(process_chunk, tasks), 1):
            stats[res[0]] = res[1:]
            if n % 20 == 0 or n == len(tasks):
                print(f"  parsed {n}/{len(tasks)} shards  ({time.time()-t0:.0f}s)", flush=True)

    n_total = sum(v[0] for v in stats.values())
    n_kept = sum(v[1] for v in stats.values())
    n_bad_char = sum(v[2] for v in stats.values())
    n_bad_frame = sum(v[3] for v in stats.values())
    n_too_long = sum(v[4] for v in stats.values())
    print(f"[{args.domain}] parsed {n_total} records, {n_kept} passed filters "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- global exact deduplication ----
    shard_ids = sorted(stats)
    hash_parts = [
        np.fromfile(os.path.join(args.tmp_dir, f"shard_{i:05d}.hash"), dtype=np.uint64)
        for i in shard_ids
    ]
    all_hashes = np.concatenate(hash_parts) if hash_parts else np.empty(0, np.uint64)
    del hash_parts
    assert len(all_hashes) == n_kept, (len(all_hashes), n_kept)

    uniq, first_idx, counts = np.unique(all_hashes, return_index=True, return_counts=True)
    keep_mask = np.zeros(n_kept, dtype=bool)
    keep_mask[first_idx] = True
    dup_count = np.ones(n_kept, dtype=np.int32)
    dup_count[first_idx] = counts
    n_unique = len(uniq)
    print(f"[{args.domain}] exact-dedup: {n_kept} -> {n_unique} "
          f"({n_kept - n_unique} duplicates removed, {time.time()-t0:.0f}s)", flush=True)
    del all_hashes, uniq, first_idx, counts

    # ---- pass 2: write the final cds.fasta.gz + meta.tsv.gz in parallel ----
    # Each shard is compressed into its own gzip member by one worker, and the
    # members are concatenated at the end.  Concatenated gzip members form a valid
    # gzip stream (zcat/pigz/python gzip all decompress the whole thing), which
    # lets pass 2 use every core too -- single-threaded gzip takes hours on
    # eukaryote.
    tax = load_taxonomy(args.species_csv)
    np.save(os.path.join(args.tmp_dir, "keep_mask.npy"), keep_mask)
    np.save(os.path.join(args.tmp_dir, "dup_count.npy"), dup_count)

    offsets = {}
    acc = 0
    for i in shard_ids:
        offsets[i] = acc
        acc += stats[i][1]
    assert acc == n_kept

    emit_tasks = [(i, offsets[i], stats[i][1], args.tmp_dir) for i in shard_ids]
    written = 0
    unmatched = set()
    with mp.Pool(args.procs, initializer=_emit_init, initargs=(tax,)) as pool:
        for n, (i, w, unm) in enumerate(pool.imap_unordered(emit_shard, emit_tasks), 1):
            written += w
            unmatched |= unm
            if n % 20 == 0 or n == len(emit_tasks):
                print(f"  emitted {n}/{len(emit_tasks)} shards, {written} seqs "
                      f"({time.time()-t0:.0f}s)", flush=True)

    assert written == n_unique, (written, n_unique)

    fa_final = os.path.join(args.out_dir, "cds.fasta.gz")
    meta_final = os.path.join(args.out_dir, "meta.tsv.gz")
    header_gz = os.path.join(args.tmp_dir, "meta_header.gz")
    with gzip.open(header_gz, "wt") as h:
        h.write("\t".join(META_COLS + ["dup_count", "taxid", "superkingdom",
                                       "phylum", "species"]) + "\n")

    def concat(dst, parts):
        with open(dst, "wb") as out:
            for p in parts:
                with open(p, "rb") as src:
                    shutil.copyfileobj(src, out, 1 << 22)

    concat(fa_final, [os.path.join(args.tmp_dir, f"shard_{i:05d}.keep.fa.gz")
                      for i in shard_ids])
    concat(meta_final, [header_gz] + [os.path.join(args.tmp_dir, f"shard_{i:05d}.keep.meta.gz")
                                      for i in shard_ids])
    print(f"[{args.domain}] concatenated -> {fa_final} ({time.time()-t0:.0f}s)", flush=True)

    # ---- report ----
    with open(args.report, "w") as r:
        r.write(f"# 01_extract_cds - {args.domain}\n\n")
        r.write(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        r.write(f"- raw directory: `{args.raw_dir}`\n")
        r.write(f"- input files (*_cds_from_genomic.fna): {len(files)}\n")
        r.write(f"- max_codons: {args.max_codons} (block_size 2048 - 4 special tokens)\n\n")
        r.write("| item | count | share |\n|---|---:|---:|\n")
        for label, v in [("raw CDS records", n_total),
                         ("non-ACGT characters (dropped)", n_bad_char),
                         ("length not a multiple of 3 (dropped)", n_bad_frame),
                         (f"more than {args.max_codons} codons (dropped)", n_too_long),
                         ("passed filters", n_kept),
                         ("after exact dedup (final)", n_unique),
                         ("removed as duplicates", n_kept - n_unique)]:
            r.write(f"| {label} | {v:,} | {v/max(n_total,1):.2%} |\n")
        r.write(f"\n- accessions not matched in the species table: {len(unmatched)}\n")
        r.write(f"- output: `{fa_final}`, `{meta_final}`\n")
    if unmatched:
        with open(args.report.replace(".md", "_unmatched_accessions.txt"), "w") as u:
            u.write("\n".join(sorted(unmatched)) + "\n")

    print(f"[{args.domain}] DONE {n_unique} sequences in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
