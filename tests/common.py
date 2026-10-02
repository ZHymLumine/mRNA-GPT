import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_ROOT = os.environ.get("MRNA_GPT_DATA") or os.path.join(ROOT, "data")
DATA = os.path.join(DATA_ROOT, "archaea")
TRAIN_LMDB = os.path.join(DATA, "train_codon.lmdb")
VAL_LMDB = os.path.join(DATA, "val_codon.lmdb")
VAL_TXT = os.path.join(DATA, "val_codon.txt.gz")
# The predecessor mRNAdesigner tree is not distributed with this repository;
# point MRNA_GPT_LEGACY_ROOT at a checkout to run the vocab-compatibility test.
LEGACY_ROOT = os.environ.get("MRNA_GPT_LEGACY_ROOT") or os.path.join(ROOT, "legacy")
LEGACY_VOCAB = os.path.join(LEGACY_ROOT, "tokenizer", "vocab.txt")

HAVE_DATA = os.path.exists(VAL_LMDB) and os.path.exists(VAL_LMDB + ".lengths.npy")
HAVE_LEGACY_VOCAB = os.path.exists(LEGACY_VOCAB)
