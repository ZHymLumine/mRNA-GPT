"""The one definition of the real high- and low-property reference genes.

Both sft/real_gene_reference.py and sft/evaluate_host_and_realism.py used to
carve "the top and bottom quartile of the held-out TEST split" out of the same
CSV, and they disagreed.  One sorted ascending and took rows[-q:] as high, the
other sorted descending and took rows[:k]; with ties at the quartile boundary
the two pick different genes.  On the bacterial task that is not a rounding
detail: its TEST split has 939 genes but only six distinct measured values,
with 255 tied at the bottom and 303 tied at the top, so a rank-based cut of 234
lands inside both tie blocks and the two implementations shared only 165 of
their 234 "high" genes.  The descriptor bands of Supplementary Figures S2 and
S6 came from one selection and the divergence denominator of Figure 5 from the
other.

The rule here is by value, not by rank: a gene is in the high set if its
measured value is at or above the 75th percentile of the split, and in the low
set if it is at or below the 25th percentile.  Every tied gene is therefore
kept, the result cannot depend on sort order or on the row order of the CSV,
and on a dataset whose measurements are a six-level ordinal it does not have to
pretend it can rank genes that were measured as equal.

For the two continuous properties this changes almost nothing (stability 1028
-> 1029 high genes, fungal transcript expression 248 -> 250); for bacteria
protein expression it takes all 303 top-scoring genes instead of an arbitrary
234 of them.
"""
from __future__ import annotations

import csv


def split_rows(rows: list[dict], quantile: float = 0.25,
               value_key: str = "Value") -> tuple[list[dict], list[dict], dict]:
    """(high, low, provenance) rows of a property TEST split.

    The cut is taken by rank from each tail and then widened to include every
    gene tied with the boundary value.  Taking the rank from each tail keeps
    the two sides symmetric, and on a split with no tie at the boundary the
    result is exactly the k highest and k lowest genes -- the same set the
    rank-based rule produced, so this change is a no-op wherever the old rule
    was already well defined.  It differs only where a tie makes the old rule
    arbitrary, which is the bacterial task and only the bacterial task.
    """
    vals = sorted(float(r[value_key]) for r in rows)
    n = len(vals)
    if n == 0:
        raise ValueError("no rows")
    k = max(1, int(quantile * n))
    lo_cut = vals[k - 1]            # k-th smallest
    hi_cut = vals[n - k]            # k-th largest
    high = [r for r in rows if float(r[value_key]) >= hi_cut]
    low = [r for r in rows if float(r[value_key]) <= lo_cut]
    prov = {"rule": "rank from each tail, all ties at the boundary included",
            "quantile": quantile, "n_total": n, "rank_k": k,
            "n_high": len(high), "n_low": len(low),
            "high_threshold": hi_cut, "low_threshold": lo_cut}
    return high, low, prov


def load_split(csv_path: str, quantile: float = 0.25
               ) -> tuple[list[dict], list[dict], dict]:
    """Read a property TEST csv and return its (high, low) reference rows.

    Rows without a sequence are dropped first, so the two callers cannot
    disagree about whether to count them.
    """
    rows = [r for r in csv.DictReader(open(csv_path)) if r.get("Sequence")]
    return split_rows(rows, quantile)
