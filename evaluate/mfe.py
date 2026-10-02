"""Minimum free energy (MFE) via ViennaRNA's RNAfold.

RNAfold is taken from $PATH (the reported numbers used ViennaRNA 2.4.7); set
RNAFOLD_BIN when it is installed somewhere that is not on PATH, e.g. inside a
dedicated conda env.
"""
from __future__ import annotations

import os
import re
import subprocess

RNAFOLD_BIN = os.environ.get("RNAFOLD_BIN", "RNAfold")

_MFE_RE = re.compile(r"\(\s*(-?\d+\.\d+)\)\s*$")


def calculate_mfe_batch(seqs: list[str], ids: list[str] | None = None,
                        rnafold_bin: str = RNAFOLD_BIN) -> dict[str, dict]:
    """seqs: RNA sequences (U alphabet). Returns {id: {"mfe": float,
    "structure": str, "mfe_per_nt": float}}. One RNAfold process for the whole
    batch (FASTA-piped), not one process per sequence."""
    ids = ids or [str(i) for i in range(len(seqs))]
    assert len(ids) == len(seqs)
    fasta = "".join(f">{i}\n{s}\n" for i, s in zip(ids, seqs))
    # RNAfold's per-sequence cost grows with length (not just sequence count),
    # so the timeout must scale with total nucleotides, not len(seqs) -- a
    # batch of long CDS (>1kb) can need far more than 1s/sequence.
    total_nt = sum(len(s) for s in seqs)
    proc = subprocess.run([rnafold_bin, "--noPS"], input=fasta, capture_output=True,
                          text=True, timeout=max(120, total_nt // 50))
    if proc.returncode != 0:
        raise RuntimeError(f"RNAfold failed: {proc.stderr}")

    out = {}
    lines = proc.stdout.strip().split("\n")
    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            i += 1
            continue
        sid = lines[i][1:].strip()
        seq_line = lines[i + 1]
        struct_line = lines[i + 2]
        m = _MFE_RE.search(struct_line)
        if not m:
            raise RuntimeError(f"could not parse RNAfold output line: {struct_line!r}")
        mfe = float(m.group(1))
        structure = struct_line[:struct_line.rindex("(")].strip()
        out[sid] = {"mfe": mfe, "structure": structure,
                    "mfe_per_nt": mfe / max(len(seq_line), 1)}
        i += 3
    missing = set(ids) - set(out)
    if missing:
        raise RuntimeError(f"RNAfold output missing entries for: {missing}")
    return out


def calculate_mfe(seq: str, rnafold_bin: str = RNAFOLD_BIN) -> dict:
    return calculate_mfe_batch([seq], ["0"], rnafold_bin)["0"]
