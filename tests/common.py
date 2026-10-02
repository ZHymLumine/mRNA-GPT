import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_ROOT = os.environ.get("MRNA_GPT_DATA") or os.path.join(ROOT, "data")
DATA = os.path.join(DATA_ROOT, "archaea")
TRAIN_LMDB = os.path.join(DATA, "train_codon.lmdb")
VAL_LMDB = os.path.join(DATA, "val_codon.lmdb")
VAL_TXT = os.path.join(DATA, "val_codon.txt.gz")
HAVE_DATA = os.path.exists(VAL_LMDB) and os.path.exists(VAL_LMDB + ".lengths.npy")
