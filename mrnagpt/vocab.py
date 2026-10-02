"""68-token codon vocabulary, the legacy-LMDB remap LUT, and the genetic code.

The published model used a 69-token vocabulary carrying ``[CLS]`` and ``[MASK]``.
Both are artifacts of bidirectional encoders (BERT's MLM / classification heads)
and have no role in an autoregressive decoder, so they are dropped here.  The
single ``[SEP]`` that previously served as both start and end marker is split
into explicit ``[BOS]`` / ``[EOS]``.

The on-disk LMDBs still use the old encoding; :func:`remap_entry` converts an
entry at read time, so no data regeneration is needed.
"""
from __future__ import annotations

import itertools

import numpy as np

# --------------------------------------------------------------------------- #
# new vocabulary
# --------------------------------------------------------------------------- #
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
CODON0 = 4

SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[BOS]", "[EOS]")
# itertools.product over "ACGU" yields exactly the alphabetical order used by the
# published vocab.txt (AAA, AAC, AAG, AAU, ACA, ...), so codon k keeps its
# relative position and only shifts by the one removed special token.
CODONS = tuple("".join(c) for c in itertools.product("ACGU", repeat=3))
ID2TOK = SPECIAL_TOKENS + CODONS
TOK2ID = {t: i for i, t in enumerate(ID2TOK)}
VOCAB_SIZE = len(ID2TOK)

assert len(CODONS) == 64 and VOCAB_SIZE == 68

# --------------------------------------------------------------------------- #
# legacy vocabulary (what the LMDBs on disk contain)
# --------------------------------------------------------------------------- #
OLD_PAD, OLD_UNK, OLD_CLS, OLD_SEP, OLD_MASK, OLD_CODON0 = 0, 1, 2, 3, 4, 5
OLD_VOCAB_SIZE = 69


def build_remap_lut() -> np.ndarray:
    """256-entry uint8 LUT mapping old token ids to new ones.

    Intended for the slice ``raw[1:-1]``; see :func:`remap_entry`.  Anything that
    is not a codon or ``[SEP]``/``[UNK]`` maps to ``[UNK]`` so that a format
    violation surfaces as UNK rather than silently shifting the vocabulary.
    """
    lut = np.full(256, UNK_ID, dtype=np.uint8)
    lut[OLD_UNK] = UNK_ID
    lut[OLD_SEP] = EOS_ID
    old = np.arange(OLD_CODON0, OLD_CODON0 + 64)
    lut[old] = (old - 1).astype(np.uint8)
    return lut


REMAP_LUT = build_remap_lut()


def remap_entry(raw: np.ndarray, check: bool = False) -> np.ndarray:
    """``[CLS][SEP] c1..cn [SEP][SEP]`` -> ``[BOS] c1..cn [EOS]`` (length n+2).

    Slicing ``raw[1:-1]`` leaves ``[SEP] c1..cn [SEP]``, which is already the
    final length.  Both ends are the same input id but need different targets,
    which a LUT cannot express, so the LUT sends ``[SEP]`` to EOS and index 0 is
    overwritten with BOS afterwards.
    """
    if check:
        assert raw.shape[0] >= 5, f"entry too short: {raw.shape[0]}"
        assert raw[0] == OLD_CLS and raw[1] == OLD_SEP, f"bad prefix {raw[:2]}"
        assert raw[-1] == OLD_SEP and raw[-2] == OLD_SEP, f"bad suffix {raw[-2:]}"
    ids = REMAP_LUT[raw[1:-1]]
    ids[0] = BOS_ID
    return ids


def encode_codons(codons) -> np.ndarray:
    """Codon strings -> ``[BOS] ids [EOS]``."""
    out = np.empty(len(codons) + 2, dtype=np.uint8)
    out[0], out[-1] = BOS_ID, EOS_ID
    for i, c in enumerate(codons):
        out[i + 1] = TOK2ID.get(c, UNK_ID)
    return out


def decode_codons(ids) -> list[str]:
    """Token ids -> codon strings, dropping every special token."""
    return [ID2TOK[int(i)] for i in ids if int(i) >= CODON0]


def write_vocab_file(path: str) -> None:
    with open(path, "w") as fh:
        fh.write("\n".join(ID2TOK) + "\n")


# --------------------------------------------------------------------------- #
# genetic code (standard table 1, RNA alphabet)
# --------------------------------------------------------------------------- #
_AA_BLOCKS = (
    ("F", "UUU UUC"), ("L", "UUA UUG CUU CUC CUA CUG"),
    ("I", "AUU AUC AUA"), ("M", "AUG"),
    ("V", "GUU GUC GUA GUG"), ("S", "UCU UCC UCA UCG AGU AGC"),
    ("P", "CCU CCC CCA CCG"), ("T", "ACU ACC ACA ACG"),
    ("A", "GCU GCC GCA GCG"), ("Y", "UAU UAC"),
    ("H", "CAU CAC"), ("Q", "CAA CAG"),
    ("N", "AAU AAC"), ("K", "AAA AAG"),
    ("D", "GAU GAC"), ("E", "GAA GAG"),
    ("C", "UGU UGC"), ("W", "UGG"),
    ("R", "CGU CGC CGA CGG AGA AGG"), ("G", "GGU GGC GGA GGG"),
    ("*", "UAA UAG UGA"),
)

GENETIC_CODE: dict[str, str] = {
    codon: aa for aa, block in _AA_BLOCKS for codon in block.split()
}
STOP_CODONS = ("UAA", "UAG", "UGA")
SENSE_CODONS = tuple(c for c in CODONS if GENETIC_CODE[c] != "*")

AA_TO_CODONS: dict[str, tuple[str, ...]] = {
    aa: tuple(c for c in CODONS if GENETIC_CODE[c] == aa)
    for aa in sorted(set(GENETIC_CODE.values()))
}

# IUPAC ambiguity + the two 21st/22nd amino acids, so a caller can hand us any
# sequence NCBI would emit without special-casing it.
_AMBIGUOUS = {
    "X": SENSE_CODONS,
    "B": AA_TO_CODONS["D"] + AA_TO_CODONS["N"],
    "Z": AA_TO_CODONS["E"] + AA_TO_CODONS["Q"],
    "J": AA_TO_CODONS["I"] + AA_TO_CODONS["L"],
    "U": ("UGA",),   # selenocysteine
    "O": ("UAG",),   # pyrrolysine
}

SYMBOLS = tuple(sorted(AA_TO_CODONS)) + tuple(_AMBIGUOUS)
SYM2I = {s: i for i, s in enumerate(SYMBOLS)}


def codons_for_symbol(sym: str) -> tuple[str, ...]:
    if sym in AA_TO_CODONS:
        return AA_TO_CODONS[sym]
    if sym in _AMBIGUOUS:
        return _AMBIGUOUS[sym]
    raise KeyError(f"unknown residue symbol {sym!r}")


def translate(codons, stop_at_first_stop: bool = False) -> str:
    """Codon strings -> amino acids; unknown codons become 'X', stops become '*'."""
    out = []
    for c in codons:
        aa = GENETIC_CODE.get(c, "X")
        if aa == "*" and stop_at_first_stop:
            break
        out.append(aa)
    return "".join(out)
