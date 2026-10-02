import os
import shutil
import unittest

from tests.common import *  # noqa: F401,F403

# mmseqs is optional: these tests are skipped unless it is on $PATH or
# MMSEQS_BIN points at an existing binary.
_MMSEQS_BIN = os.environ.get("MMSEQS_BIN")
HAVE_MMSEQS = (shutil.which("mmseqs") is not None
               or (_MMSEQS_BIN is not None and os.path.exists(_MMSEQS_BIN)))

from evaluate.diversity import (batch_diversity, kmer_entropy, kmer_profile,  # noqa: E402
                                nearest_neighbor_identity)


class TestKmerEntropy(unittest.TestCase):
    def test_repetitive_sequence_has_zero_entropy(self):
        self.assertAlmostEqual(kmer_entropy("AAAAAAAAAAAA", k=4), 0.0)

    def test_varied_sequence_has_positive_entropy(self):
        self.assertGreater(kmer_entropy("AUGGCUAAAGGCUUUCCCAAAGGGUUUCCC", k=4), 0.0)

    def test_profile_sums_to_one(self):
        p = kmer_profile("AUGGCUAAAUAG", k=3)
        self.assertAlmostEqual(sum(p.values()), 1.0)


class TestBatchDiversity(unittest.TestCase):
    def test_identical_batch_has_zero_js_divergence(self):
        d = batch_diversity(["AUGGCUAAAUAG"] * 5, k=3)
        self.assertAlmostEqual(d["mean_pairwise_js_divergence"], 0.0)
        self.assertAlmostEqual(d["exact_duplicate_fraction"], 1.0 - 1 / 5)

    def test_distinct_batch_has_zero_duplicate_fraction(self):
        d = batch_diversity(["AUGGCUAAAUAG", "AUGCCCGGGUAA", "AUGAAAUUUUAG"], k=3)
        self.assertEqual(d["exact_duplicate_fraction"], 0.0)
        self.assertGreater(d["mean_pairwise_js_divergence"], 0.0)

    def test_pair_sampling_cap(self):
        seqs = [f"AUG{'GCU' * (i % 3 + 1)}UAA" for i in range(50)]
        d = batch_diversity(seqs, k=3, max_pairs=10)
        self.assertEqual(d["n_pairs_sampled"], 10)


@unittest.skipUnless(HAVE_MMSEQS, "mmseqs2 not available")
class TestNearestNeighbor(unittest.TestCase):
    def setUp(self):
        self.ref = {"r1": "ATGGCTAAAGGCTTTCCCAAAGGGTTTCCCTAA",
                    "r2": "ATGCCCGGGAAATTTCCCGGGAAATTTTAA"}

    def test_exact_match_is_100_percent(self):
        r = nearest_neighbor_identity({"q1": self.ref["r1"]}, self.ref)
        self.assertEqual(r["q1"]["best_identity"], 100.0)
        self.assertEqual(r["q1"]["best_hit"], "r1")

    def test_unrelated_sequence_has_no_hit(self):
        qry = {"q3": "ATGAAATTTGGGCCCAAATTTGGGCCCAAATAA"}
        r = nearest_neighbor_identity(qry, self.ref)
        self.assertEqual(r["q3"]["best_hit"], None)
        self.assertEqual(r["q3"]["best_identity"], 0.0)

    def test_every_query_gets_a_result(self):
        qry = {"a": self.ref["r1"], "b": "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"}
        r = nearest_neighbor_identity(qry, self.ref)
        self.assertEqual(set(r), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
