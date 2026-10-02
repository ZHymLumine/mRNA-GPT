"""Codon-usage evaluators: GC/GC3, CAI, tAI.

Independent of the mrnagpt training package on purpose -- these operate on
plain codon lists (RNA alphabet) so they apply equally to training data,
pretrained-model generations, and fine-tuned-model generations.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict

GENETIC_CODE = {
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L", "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M", "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S", "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*", "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K", "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W", "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R", "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
STOP_CODONS = ("UAA", "UAG", "UGA")

_HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# GC content
# --------------------------------------------------------------------------- #
def gc_content(codons) -> float:
    seq = "".join(codons)
    return sum(c in "GC" for c in seq) / max(len(seq), 1)


def gc3_content(codons) -> float:
    third = [c[2] for c in codons if len(c) == 3]
    return sum(c in "GC" for c in third) / max(len(third), 1)


# --------------------------------------------------------------------------- #
# CAI (Sharp & Li 1987): reference-set relative adaptiveness, geometric mean
# --------------------------------------------------------------------------- #
def build_cai_reference(reference_codon_lists) -> dict:
    """reference_codon_lists: iterable of codon-list sequences (the high-expression
    reference set). Returns {codon: w} with w = freq(codon) / max_freq(synonyms)."""
    counts = defaultdict(int)
    for codons in reference_codon_lists:
        for c in codons:
            aa = GENETIC_CODE.get(c)
            if aa and aa != "*":
                counts[c] += 1
    by_aa = defaultdict(dict)
    for c, n in counts.items():
        by_aa[GENETIC_CODE[c]][c] = n
    w = {}
    for aa, codon_counts in by_aa.items():
        m = max(codon_counts.values())
        for c, n in codon_counts.items():
            w[c] = n / m
    return w


def save_cai_reference(w: dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(w, fh, indent=2, sort_keys=True)


def load_cai_reference(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def cai(codons, w: dict) -> float:
    """Geometric mean of relative adaptiveness over non-stop codons with a
    reference weight (unseen/single-codon-family codons are skipped, matching
    standard practice: Met/Trp have only one codon, so w=1 trivially)."""
    log_sum, n = 0.0, 0
    for c in codons:
        if c in STOP_CODONS:
            continue
        wc = w.get(c)
        if wc is None or wc <= 0:
            continue
        log_sum += math.log(wc)
        n += 1
    if n == 0:
        return float("nan")
    return math.exp(log_sum / n)


# --------------------------------------------------------------------------- #
# tAI (dos Reis, Savva & Wernisch 2004), S. cerevisiae reference weights
# --------------------------------------------------------------------------- #
_TAI_CACHE = None


def load_tai_weights(path: str | None = None) -> dict:
    """Loads the precomputed 60-codon S. cerevisiae tAI weight vector.

    Computed once via evaluate/tai_weights.R (the reference implementation from
    https://github.com/mariodosreis/tai), fed with GtRNAdb S. cerevisiae tRNA
    gene copy numbers (evaluate/data/scer_trna_gene_counts.json). Recomputing
    at evaluation time would require R + jsonlite; this module only needs the
    resulting JSON.
    """
    global _TAI_CACHE
    if _TAI_CACHE is not None:
        return _TAI_CACHE
    path = path or os.path.join(_HERE, "data", "scer_tai_weights.json")
    with open(path) as fh:
        d = json.load(fh)
    codon_order = [c.replace("T", "U") for c in d["codon_order"]]
    _TAI_CACHE = dict(zip(codon_order, d["w"]))
    return _TAI_CACHE


def tai(codons, w: dict | None = None) -> float:
    """tAI = exp(mean(log w_i)) over non-stop, non-Met codons (dos Reis 2004;
    Met is conventionally excluded because AUG's tRNA pool is initiator-biased
    in a way the model doesn't separately account for)."""
    w = w or load_tai_weights()
    log_sum, n = 0.0, 0
    for c in codons:
        if c in STOP_CODONS or GENETIC_CODE.get(c) == "M":
            continue
        wc = w.get(c)
        if wc is None or wc <= 0:
            continue
        log_sum += math.log(wc)
        n += 1
    if n == 0:
        return float("nan")
    return math.exp(log_sum / n)
