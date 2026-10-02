"""Fetch the native (wild-type) CDS for each panel target protein.

Serves two purposes:
  - a wild-type reference row for every comparison table: what the natural coding
    sequence for this protein scores under the same evaluators as the designs
  - the starting sequence for iCodon, whose optimizer is a genetic algorithm over
    an existing CDS rather than a protein-conditioned generator

Each CDS is fetched from the EMBL/GenBank record cross-referenced by UniProt for
that accession, then translated and compared to the panel protein. A CDS whose
translation does not match is reported, not silently used: the panel sequences
come from a third-party table and need not agree residue-for-residue with the
database entry (isoforms, strain variants, signal-peptide handling).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sft.paths import RUNS                                       # noqa: E402

from mrnagpt.vocab import GENETIC_CODE, STOP_CODONS

# UniProt accession -> (panel target slug, nucleotide accession, protein_id)
TARGETS = {
    "P08667": ("Rabies_virus", "M13215", "AAA47218.1"),
    "P87671": ("Zaire_ebolavirus", "U81161", "AAC57992.1"),
    "P0DTC5": ("SARS_CoV_2", "MN908947", "QHD43419.1"),
    "P40126": ("Homo_sapiens", "D17547", "BAA04484.1"),
}
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def efetch_cds(nuc_acc: str) -> list[tuple[str, str]]:
    q = urllib.parse.urlencode({"db": "nuccore", "id": nuc_acc,
                                "rettype": "fasta_cds_na", "retmode": "text"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"{EFETCH}?{q}", timeout=90) as fh:
                text = fh.read().decode()
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))
    out, header, cur = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                out.append((header, "".join(cur)))
            header, cur = line, []
        elif line.strip():
            cur.append(line.strip())
    if header is not None:
        out.append((header, "".join(cur)))
    return out


def translate(dna: str) -> str:
    rna = dna.upper().replace("T", "U")
    cod = [rna[i:i + 3] for i in range(0, len(rna) - len(rna) % 3, 3)]
    body = cod[:-1] if cod and cod[-1] in STOP_CODONS else cod
    return "".join(GENETIC_CODE.get(c, "?") for c in body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-csv", default="data/protein_sequences.csv")
    ap.add_argument("--out-fasta", default=str(RUNS / "fungal_sft" /
                                                "generation_panel/baselines/native_cds.fasta"))
    ap.add_argument("--out-json", default="sft/data/native_cds.json")
    args = ap.parse_args()

    panel = {re.sub(r"[^A-Za-z0-9]+", "_", r["organism"]).strip("_"): r["sequence"].strip().upper()
             for r in csv.DictReader(open(args.panel_csv))}

    meta, records = {}, []
    for acc, (target, nuc, pid) in TARGETS.items():
        hits = efetch_cds(nuc)
        chosen = None
        for header, seq in hits:
            if pid.split(".")[0] in header:
                chosen = (header, seq)
                break
        if chosen is None:
            raise SystemExit(f"{target}: protein_id {pid} not found among "
                             f"{len(hits)} CDS features of {nuc}")
        header, dna = chosen
        prot = translate(dna)
        want = panel[target]
        n = min(len(prot), len(want))
        ident = 100.0 * sum(a == b for a, b in zip(prot, want)) / max(len(want), 1)
        meta[target] = {"uniprot": acc, "nucleotide": nuc, "protein_id": pid,
                        "cds_nt": len(dna), "translated_aa": len(prot),
                        "panel_aa": len(want), "exact_match": prot == want,
                        "residue_identity_pct": ident}
        print(f"{target}: {nuc}/{pid}  CDS {len(dna)} nt -> {len(prot)} aa "
              f"(panel {len(want)} aa)  exact={prot == want}  identity {ident:.2f}%", flush=True)
        records.append((target, dna.upper().replace("T", "U")))

    with open(args.out_fasta, "w") as fh:
        for target, rna in records:
            fh.write(f">native_cds|{target}|0 n_codon={len(rna)//3}\n{rna}\n")
    with open(args.out_json, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {args.out_fasta} and {args.out_json}")


if __name__ == "__main__":
    main()
