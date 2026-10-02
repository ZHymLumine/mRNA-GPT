"""Novelty and diversity evaluators for generated CDS.

Novelty (memorization check): nearest-neighbor %identity of each generated
sequence against the training set, via MMseqs2 -- the same homology-search
machinery already validated for the pretraining-leakage work, reused here for
its intended purpose (fast approximate nearest-neighbor search) rather than
re-implemented as slow pairwise edit distance.

Diversity (mode-collapse check): k-mer composition diversity within a generated
batch -- Shannon entropy per sequence, and pairwise Jensen-Shannon divergence /
cosine distance between sequences' k-mer profiles. Pure Python, O(n^2) k-mer
vector comparisons rather than O(n^2) edit distance, which is what makes a batch
of ~1,000 sequences tractable without leaving Python.
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
from collections import Counter

# Resolved lazily (not at import time) so MMSEQS_BIN can be set after import.
# The default is a bare "mmseqs" taken from $PATH; set MMSEQS_BIN when the
# binary lives somewhere that is not on PATH (e.g. inside a conda env).
_DEFAULT_MMSEQS = "mmseqs"


def _mmseqs_bin() -> str:
    import shutil
    env = os.environ.get("MMSEQS_BIN")
    if env:
        return env
    if shutil.which("mmseqs"):
        return "mmseqs"
    return _DEFAULT_MMSEQS


# --------------------------------------------------------------------------- #
# novelty: nearest-neighbor identity against a reference set (e.g. training data)
# --------------------------------------------------------------------------- #
def write_fasta(seqs: dict[str, str], path: str) -> None:
    with open(path, "w") as fh:
        for sid, s in seqs.items():
            fh.write(f">{sid}\n{s}\n")


def nearest_neighbor_identity(query: dict[str, str], reference: dict[str, str], *,
                              min_seq_id: float = 0.0, threads: int = 4,
                              split_memory_limit: str = "40G",
                              mmseqs_bin: str | None = None) -> dict[str, dict]:
    """For each query sequence, the single best %identity hit in `reference`
    (DNA alphabet expected; U is converted to T automatically). Returns
    {query_id: {"best_identity": float in [0,100], "best_hit": ref_id or None}}
    -- a query with no hit above `min_seq_id` has best_identity=0, best_hit=None,
    which is exactly the signal of a genuinely novel sequence.
    """
    mmseqs_bin = mmseqs_bin or _mmseqs_bin()

    def to_dna(s):
        return s.upper().replace("U", "T")

    with tempfile.TemporaryDirectory() as d:
        qfa, rfa = os.path.join(d, "q.fasta"), os.path.join(d, "r.fasta")
        write_fasta({k: to_dna(v) for k, v in query.items()}, qfa)
        write_fasta({k: to_dna(v) for k, v in reference.items()}, rfa)
        qdb, rdb = os.path.join(d, "qdb"), os.path.join(d, "rdb")
        res, tmp = os.path.join(d, "res"), os.path.join(d, "tmp")
        for cmd in (
            [mmseqs_bin, "createdb", qfa, qdb, "--dbtype", "2", "-v", "1"],
            [mmseqs_bin, "createdb", rfa, rdb, "--dbtype", "2", "-v", "1"],
            [mmseqs_bin, "search", qdb, rdb, res, tmp, "--search-type", "3",
             "--strand", "1", "-s", "7.5", "--min-seq-id", str(min_seq_id),
             "-c", "0.8", "--cov-mode", "0", "-e", "10.0", "--max-seqs", "1",
             "--split-memory-limit", split_memory_limit, "--threads", str(threads),
             "-v", "1"],
        ):
            subprocess.run(cmd, check=True, capture_output=True)
        alis = os.path.join(d, "alis.tsv")
        subprocess.run([mmseqs_bin, "convertalis", qdb, rdb, res, alis,
                        "--format-output", "query,target,pident", "-v", "1"], check=True)
        best = {}
        with open(alis) as fh:
            for line in fh:
                q, t, pid = line.strip().split("\t")
                pid = float(pid)
                if q not in best or pid > best[q]["best_identity"]:
                    best[q] = {"best_identity": pid, "best_hit": t}

    out = {}
    for qid in query:
        out[qid] = best.get(qid, {"best_identity": 0.0, "best_hit": None})
    return out


# --------------------------------------------------------------------------- #
# diversity: k-mer composition
# --------------------------------------------------------------------------- #
def kmer_counts(seq: str, k: int) -> Counter:
    return Counter(seq[i:i + k] for i in range(len(seq) - k + 1))


def kmer_entropy(seq: str, k: int = 4) -> float:
    """Shannon entropy (bits) of the k-mer frequency distribution within one
    sequence. A collapsed/repetitive sequence has low entropy; a sequence with
    natural, non-repetitive composition is close to the k-mer alphabet's max
    entropy (log2(4^k))."""
    c = kmer_counts(seq, k)
    n = sum(c.values())
    if n == 0:
        return 0.0
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def kmer_profile(seq: str, k: int) -> dict[str, float]:
    c = kmer_counts(seq, k)
    n = sum(c.values()) or 1
    return {kmer: v / n for kmer, v in c.items()}


def _js_divergence(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    m = {kk: 0.5 * (p.get(kk, 0.0) + q.get(kk, 0.0)) for kk in keys}

    def kl(a, b):
        s = 0.0
        for kk in keys:
            av = a.get(kk, 0.0)
            if av > 0:
                s += av * math.log2(av / b[kk])
        return s

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def batch_diversity(seqs: list[str], k: int = 4, max_pairs: int = 200_000,
                    seed: int = 0) -> dict:
    """Per-sequence k-mer entropy, plus mean pairwise Jensen-Shannon divergence
    between sequences' k-mer profiles (0 = identical composition, up to 1 bit
    for maximally different). A batch suffering mode collapse shows a low mean
    pairwise JS divergence -- generated sequences all "look the same" compositionally,
    even if not literally identical strings."""
    profiles = [kmer_profile(s, k) for s in seqs]
    entropies = [kmer_entropy(s, k) for s in seqs]

    n = len(seqs)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        import random
        random.Random(seed).shuffle(pairs)
        pairs = pairs[:max_pairs]

    js = [_js_divergence(profiles[i], profiles[j]) for i, j in pairs]
    unique_fraction = len(set(seqs)) / max(n, 1)

    return {
        "n_seqs": n,
        "mean_kmer_entropy": sum(entropies) / max(n, 1),
        "mean_pairwise_js_divergence": sum(js) / max(len(js), 1) if js else float("nan"),
        "n_pairs_sampled": len(pairs),
        "exact_duplicate_fraction": 1.0 - unique_fraction,
    }
