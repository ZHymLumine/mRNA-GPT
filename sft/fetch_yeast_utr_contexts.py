"""Fetch fixed 5'/3' UTR contexts for S. cerevisiae genes used in the UTR-swap test.

S. cerevisiae UTRs are not annotated in Ensembl (cDNA == CDS for these genes) nor
served by SGD, so the flanking genomic sequence is used as the UTR proxy, 200 nt on
each side -- the same length as the verified real ADH1 context already in
sft/data/adh1_utr_context.py, so all contexts are directly comparable.

Each fetch is checked: the CDS carved out of the expanded genomic sequence must
start with ATG and end in a stop codon, which also confirms Ensembl returned the
gene in transcribed orientation for minus-strand genes.

Gene sets:
  destabilizing -- PIR1, RNR1, ERG5, HTB1, HHF1 (from Rahaman et al. 2023)
  stable        -- ADH1, TDH3, PGK1, ENO2 (canonical highly expressed yeast genes,
                   included so the test is a contrast and not a single-arm reading)
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

FLANK = 200
STOPS = {"TAA", "TAG", "TGA"}

GENES = {
    "PIR1": ("YKL164C", "destabilizing"),
    "RNR1": ("YER070W", "destabilizing"),
    "ERG5": ("YMR015C", "destabilizing"),
    "HTB1": ("YDR224C", "destabilizing"),
    "HHF1": ("YBR009C", "destabilizing"),
    "ADH1": ("YOL086C", "stable"),
    "TDH3": ("YGR192C", "stable"),
    "PGK1": ("YCR012W", "stable"),
    "ENO2": ("YHR174W", "stable"),
}


def get(url: str) -> dict:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as fh:
                return json.load(fh)
        except Exception as exc:                      # transient REST failures
            if attempt == 3:
                raise
            print(f"  retry {attempt + 1} after {type(exc).__name__}", flush=True)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch(gene_id: str) -> dict:
    base = "https://rest.ensembl.org/sequence/id/"
    exp = get(f"{base}{gene_id}?type=genomic;expand_5prime={FLANK};"
              f"expand_3prime={FLANK};content-type=application/json")["seq"].upper()
    cds = get(f"{base}{gene_id}?type=cds;content-type=application/json")["seq"].upper()
    carved = exp[FLANK:len(exp) - FLANK]
    ok = carved == cds and cds[:3] == "ATG" and cds[-3:] in STOPS
    return {"utr5": exp[:FLANK], "utr3": exp[len(exp) - FLANK:], "cds": cds,
            "cds_len": len(cds), "verified": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sft/data/yeast_utr_contexts.json")
    args = ap.parse_args()

    out = {}
    for name, (gid, group) in GENES.items():
        rec = fetch(gid)
        rec.update(gene=name, gene_id=gid, group=group, flank_nt=FLANK)
        out[name] = rec
        print(f"{name} ({gid}, {group}): CDS {rec['cds_len']} nt, "
              f"start/stop+carve check {'OK' if rec['verified'] else 'FAILED'}", flush=True)
    bad = [n for n, r in out.items() if not r["verified"]]
    if bad:
        raise SystemExit(f"failed verification: {bad} -- do not use these contexts")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
