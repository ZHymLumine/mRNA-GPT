import math
import unittest

from tests.common import *  # noqa: F401,F403 (sys.path setup)
from evaluate.codon_metrics import (GENETIC_CODE, STOP_CODONS, build_cai_reference,
                                    cai, gc3_content, gc_content, load_tai_weights, tai)


class TestGC(unittest.TestCase):
    def test_all_gc(self):
        self.assertAlmostEqual(gc_content(["GGG", "CCC"]), 1.0)
        self.assertAlmostEqual(gc3_content(["GGG", "CCC"]), 1.0)

    def test_all_at(self):
        self.assertAlmostEqual(gc_content(["AAA", "UUU"]), 0.0)

    def test_gc3_only_third_position(self):
        # third position all G/C, first two positions all A/U
        self.assertAlmostEqual(gc3_content(["AUG", "AUC"]), 1.0)
        self.assertLess(gc_content(["AUG", "AUC"]), 1.0)


class TestCAI(unittest.TestCase):
    def test_reference_codon_gets_weight_one(self):
        ref = build_cai_reference([["GCC"] * 9 + ["GCU"]])
        self.assertEqual(ref["GCC"], 1.0)
        self.assertAlmostEqual(ref["GCU"], 1 / 9)

    def test_single_codon_family_has_weight_one(self):
        ref = build_cai_reference([["AUG", "UGG"]])   # Met, Trp: only one codon each
        self.assertEqual(ref["AUG"], 1.0)
        self.assertEqual(ref["UGG"], 1.0)

    def test_optimal_sequence_scores_near_one(self):
        ref = build_cai_reference([["GCC", "AAA", "GGC"] * 20])
        self.assertAlmostEqual(cai(["GCC", "AAA", "GGC"], ref), 1.0, places=6)

    def test_range(self):
        # GCC/AAA/GGC dominate the reference; GCU/AAG/GGA are minority synonyms
        ref = build_cai_reference([["GCC"] * 9 + ["GCU"] + ["AAA"] * 9 + ["AAG"]
                                   + ["GGC"] * 9 + ["GGA"]])
        v = cai(["GCU", "AAG", "GGA"], ref)
        self.assertTrue(0.0 < v < 1.0, v)
        self.assertAlmostEqual(cai(["GCC", "AAA", "GGC"], ref), 1.0, places=6)

    def test_stop_codon_excluded(self):
        ref = build_cai_reference([["GCC"] * 10])
        a = cai(["GCC", "GCC"], ref)
        b = cai(["GCC", "GCC", "UAA"], ref)
        self.assertAlmostEqual(a, b)


class TestTAI(unittest.TestCase):
    def test_weight_table_shape(self):
        w = load_tai_weights()
        # 64 codons - 3 stops - 1 Met = 60
        self.assertEqual(len(w), 60)
        for c in w:
            self.assertIn(c, GENETIC_CODE)
            self.assertNotIn(c, STOP_CODONS)
            self.assertNotEqual(GENETIC_CODE[c], "M")

    def test_weights_bounded(self):
        w = load_tai_weights()
        for v in w.values():
            self.assertGreater(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_met_and_stop_excluded_from_score(self):
        w = load_tai_weights()
        a = tai(["GCC"], w)
        b = tai(["GCC", "AUG", "UAA"], w)
        self.assertAlmostEqual(a, b)

    def test_range(self):
        w = load_tai_weights()
        v = tai(["AAA", "GAC", "GAA", "AUC"], w)
        self.assertTrue(0.0 < v <= 1.0)
        self.assertFalse(math.isnan(v))


if __name__ == "__main__":
    unittest.main()
