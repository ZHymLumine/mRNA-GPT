import gzip
import unittest

import numpy as np

from tests.common import HAVE_DATA, VAL_LMDB, VAL_TXT
from mrnagpt import vocab as v


class TestVocab(unittest.TestCase):
    def test_sizes_and_ids(self):
        self.assertEqual(v.VOCAB_SIZE, 68)
        self.assertEqual(v.ID2TOK[:4], ("[PAD]", "[UNK]", "[BOS]", "[EOS]"))
        self.assertEqual((v.PAD_ID, v.UNK_ID, v.BOS_ID, v.EOS_ID), (0, 1, 2, 3))
        self.assertEqual(len(v.CODONS), 64)
        self.assertEqual(len(set(v.CODONS)), 64)

    def test_codon_order_matches_stored_encoding(self):
        """Stored id k must map to model id k-1; anything else corrupts data."""
        self.assertEqual(v.STORED_VOCAB_SIZE, v.VOCAB_SIZE + 1)
        lut = v.build_remap_lut()
        for k, codon in enumerate(v.CODONS):
            stored_id = v.STORED_CODON0 + k
            self.assertEqual(int(lut[stored_id]), v.TOK2ID[codon])
            self.assertEqual(int(lut[stored_id]), stored_id - 1)

    def test_lut(self):
        lut = v.build_remap_lut()
        self.assertEqual(lut.dtype, np.uint8)
        self.assertEqual(lut.shape, (256,))
        self.assertEqual(int(lut[v.STORED_SEP]), v.EOS_ID)
        self.assertEqual(int(lut[v.STORED_UNK]), v.UNK_ID)
        for old_id in range(5, 69):
            self.assertEqual(int(lut[old_id]), old_id - 1)
        for bad in (v.STORED_PAD, v.STORED_CLS, v.STORED_MASK, 200, 255):
            self.assertEqual(int(lut[bad]), v.UNK_ID)

    def test_genetic_code(self):
        self.assertEqual(set(v.GENETIC_CODE), set(v.CODONS))
        self.assertEqual(len(v.SENSE_CODONS), 61)
        self.assertEqual(set(v.STOP_CODONS), {"UAA", "UAG", "UGA"})
        # every codon belongs to exactly one amino acid block
        total = sum(len(cs) for cs in v.AA_TO_CODONS.values())
        self.assertEqual(total, 64)
        self.assertEqual(v.GENETIC_CODE["AUG"], "M")
        self.assertEqual(v.GENETIC_CODE["UGG"], "W")
        self.assertEqual(len(v.AA_TO_CODONS["L"]), 6)
        self.assertEqual(len(v.AA_TO_CODONS["S"]), 6)
        self.assertEqual(len(v.AA_TO_CODONS["R"]), 6)
        for sym in v.SYMBOLS:
            self.assertGreater(len(v.codons_for_symbol(sym)), 0)

    def test_encode_decode_roundtrip(self):
        codons = ["AUG", "GCU", "AAA", "UAA"]
        ids = v.encode_codons(codons)
        self.assertEqual(int(ids[0]), v.BOS_ID)
        self.assertEqual(int(ids[-1]), v.EOS_ID)
        self.assertEqual(v.decode_codons(ids), codons)

    @unittest.skipUnless(HAVE_DATA, "archaea LMDB not available")
    def test_lut_roundtrip_against_text(self):
        """Decode LMDB entries through the LUT and compare to *_codon.txt.gz."""
        import lmdb
        n = 500
        lines = []
        with gzip.open(VAL_TXT, "rt") as fh:
            for i, line in enumerate(fh):
                lines.append(line.split())
                if i + 1 >= n:
                    break
        env = lmdb.open(VAL_LMDB, subdir=False, readonly=True, lock=False)
        with env.begin(buffers=True) as txn:
            for i in range(n):
                raw = np.frombuffer(txn.get(b"%d" % i), dtype=np.uint8)
                self.assertEqual(int(raw[0]), v.STORED_CLS)
                self.assertEqual(int(raw[1]), v.STORED_SEP)
                self.assertEqual(int(raw[-1]), v.STORED_SEP)
                self.assertEqual(int(raw[-2]), v.STORED_SEP)
                ids = v.remap_entry(raw, check=True)
                self.assertEqual(int(ids[0]), v.BOS_ID)
                self.assertEqual(int(ids[-1]), v.EOS_ID)
                self.assertEqual(len(ids), len(lines[i]) + 2)
                self.assertEqual(v.decode_codons(ids), lines[i])
        env.close()


if __name__ == "__main__":
    unittest.main()
