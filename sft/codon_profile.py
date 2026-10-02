"""The codon-profile metric, in one place.

family_profile / js_divergence / score_group were reimplemented three times
while Figure 5 was being made consistent, which is the same failure mode as the
two quartile rules that disagreed: a metric with more than one implementation
eventually has more than one value.  Everything that scores synonymous codon
usage imports from here.

Deliberately free of `evaluate.codon_metrics`, so this module can be imported
under a plain interpreter; the repository's namespace package `evaluate/` is
shadowed by an installed distribution of the same name.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict

from mrnagpt.vocab import GENETIC_CODE

PSEUDO = 0.5   # Laplace-style smoothing so an unused codon never gives -inf


def family_profile(codon_lists) -> dict[str, dict[str, float]]:
    """p(codon | amino acid), smoothed, over a set of sequences."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for codons in codon_lists:
        for c in codons:
            aa = GENETIC_CODE.get(c)
            if aa and aa != "*":
                counts[aa][c] += 1
    families = defaultdict(list)
    for c, aa in GENETIC_CODE.items():
        if aa != "*":
            families[aa].append(c)
    prof = {}
    for aa, members in families.items():
        tot = sum(counts[aa][c] for c in members) + PSEUDO * len(members)
        prof[aa] = {c: (counts[aa][c] + PSEUDO) / tot for c in members}
    return prof


def _kl(p: dict, q: dict) -> float:
    return sum(pv * math.log2(pv / q[c]) for c, pv in p.items() if pv > 0)


def js_divergence(p: dict, q: dict) -> float:
    m = {c: 0.5 * (p[c] + q[c]) for c in p}
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def score_group(codon_lists, prof_high, prof_low) -> dict:
    """Usage-weighted JS to BOTH real profiles, and mean per-codon LLR.

    Reporting the distance to the low-property profile as well is what separates
    "learned to look like this host" from "learned to look high-property": a
    model fine-tuned on the bottom quartile sits in the same host, so if the two
    distances did not swap for it, the metric would be measuring the species and
    not the property it claims to.
    """
    prof = family_profile(codon_lists)
    usage = Counter()
    for codons in codon_lists:
        for c in codons:
            aa = GENETIC_CODE.get(c)
            if aa and aa != "*":
                usage[aa] += 1
    multi = [aa for aa in prof if len(prof[aa]) > 1]        # Met/Trp carry no choice
    tot = sum(usage[aa] for aa in multi) or 1
    js = sum(usage[aa] * js_divergence(prof[aa], prof_high[aa]) for aa in multi) / tot
    js_low = sum(usage[aa] * js_divergence(prof[aa], prof_low[aa]) for aa in multi) / tot

    llr_vals = []
    for codons in codon_lists:
        vals = [math.log(prof_high[GENETIC_CODE[c]][c] / prof_low[GENETIC_CODE[c]][c])
                for c in codons
                if GENETIC_CODE.get(c) and GENETIC_CODE[c] != "*"
                and len(prof_high[GENETIC_CODE[c]]) > 1]
        if vals:
            llr_vals.append(statistics.mean(vals))
    return {"js_to_high": js, "js_to_low": js_low, "js_high_minus_low": js - js_low,
            "llr_high_low": statistics.mean(llr_vals) if llr_vals else float("nan"),
            "n": len(codon_lists)}


