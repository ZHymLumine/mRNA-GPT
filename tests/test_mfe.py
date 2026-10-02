import os
import unittest

from tests.common import *  # noqa: F401,F403
from evaluate.mfe import RNAFOLD_BIN, calculate_mfe, calculate_mfe_batch

HAVE_RNAFOLD = os.path.exists(RNAFOLD_BIN)


@unittest.skipUnless(HAVE_RNAFOLD, "RNAfold binary not available")
class TestMFE(unittest.TestCase):
    def test_hairpin_is_stable(self):
        # a strong hairpin should fold with negative (favourable) MFE
        r = calculate_mfe("GGGGGAAAACCCCC")
        self.assertLess(r["mfe"], -3.0)
        self.assertIn("(", r["structure"])

    def test_unstructured_sequence_near_zero(self):
        r = calculate_mfe("AUGAAAUAG")
        self.assertGreaterEqual(r["mfe"], -1.0)

    def test_batch_matches_single(self):
        seqs = ["GGGGGAAAACCCCC", "AUGAAAUAG"]
        single = [calculate_mfe(s)["mfe"] for s in seqs]
        batch = calculate_mfe_batch(seqs, ["x", "y"])
        self.assertAlmostEqual(single[0], batch["x"]["mfe"])
        self.assertAlmostEqual(single[1], batch["y"]["mfe"])

    def test_mfe_per_nt_is_normalised_by_length(self):
        r = calculate_mfe("GGGGGAAAACCCCC")
        self.assertAlmostEqual(r["mfe_per_nt"], r["mfe"] / len("GGGGGAAAACCCCC"))


if __name__ == "__main__":
    unittest.main()
