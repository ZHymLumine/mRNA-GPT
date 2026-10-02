import random
import unittest

import torch

from tests.common import *  # noqa: F401,F403
from mrnagpt.generate import (build_allowed_mask, constrained_sample, sample,
                              validate_batch, validate_cds)
from mrnagpt.model import GPT, GPTConfig
from mrnagpt.vocab import (AA_TO_CODONS, STOP_CODONS, SYM2I, SYMBOLS, TOK2ID,
                           codons_for_symbol, translate)

AAS = "ACDEFGHIKLMNPQRSTVWY"


def tiny(block_size=1024):
    torch.manual_seed(0)
    return GPT(GPTConfig(pos_encoding="rope", n_layer=2, n_head=4, n_embd=64,
                         block_size=block_size, dropout=0.0)).eval()


class TestConstrained(unittest.TestCase):
    def test_allowed_mask(self):
        m = build_allowed_mask("cpu")
        self.assertEqual(m.shape, (len(SYMBOLS), 68))
        for sym in SYMBOLS:
            row = m[SYM2I[sym]]
            self.assertEqual(int(row.sum()), len(set(codons_for_symbol(sym))))
            for c in codons_for_symbol(sym):
                self.assertTrue(bool(row[TOK2ID[c]]))
        # specials are never selectable
        self.assertEqual(int(m[:, :4].sum()), 0)

    def test_translation_is_exact_on_untrained_model(self):
        """Correctness is constructive: it must hold for a random-init model."""
        m = tiny()
        rng = random.Random(0)
        prots = ["M" + "".join(rng.choice(AAS) for _ in range(rng.randint(4, 200)))
                 for _ in range(200)]
        outs = constrained_sample(m, prots, device="cpu", temperature=0.9)
        for out, p in zip(outs, prots):
            self.assertEqual(len(out), len(p) + 1)
            self.assertEqual(translate(out[:-1]), p)
            self.assertIn(out[-1], STOP_CODONS)
            self.assertEqual(out[0], "AUG")

    def test_no_stop_when_disabled(self):
        m = tiny()
        outs = constrained_sample(m, ["MKV"], add_stop=False, device="cpu")
        self.assertEqual(len(outs[0]), 3)
        self.assertEqual(translate(outs[0]), "MKV")

    def test_force_atg_only_when_protein_starts_with_M(self):
        m = tiny()
        outs = constrained_sample(m, ["MAA", "KAA"], device="cpu")
        self.assertEqual(outs[0][0], "AUG")
        self.assertIn(outs[1][0], AA_TO_CODONS["K"])

    def test_ragged_batch(self):
        m = tiny()
        prots = ["MA", "M" + "K" * 40, "MCDEF"]
        outs = constrained_sample(m, prots, device="cpu")
        for out, p in zip(outs, prots):
            self.assertEqual(translate(out[:-1]), p)

    def test_cache_matches_no_cache(self):
        m = tiny()
        prots = ["MKVLAAG", "MDDEEFF"]
        g1 = torch.Generator().manual_seed(7)
        g2 = torch.Generator().manual_seed(7)
        a = constrained_sample(m, prots, device="cpu", generator=g1, use_cache=True)
        b = constrained_sample(m, prots, device="cpu", generator=g2, use_cache=False)
        self.assertEqual(a, b)

    def test_sliding_window_beyond_block_size(self):
        """Proteins longer than the context window (Dystrophin is 3685 aa) still
        decode to exactly the target."""
        m = tiny(block_size=64)
        rng = random.Random(3)
        p = "M" + "".join(rng.choice(AAS) for _ in range(150))
        for mode in ("anchor", "truncate", "rope_extend"):
            with self.subTest(mode=mode):
                out = constrained_sample(m, [p], device="cpu", window_mode=mode,
                                         use_cache=False)[0]
                self.assertEqual(translate(out[:-1]), p)

    def test_chunking_preserves_order_and_correctness(self):
        """The KV cache is (B, n_head, L, head_dim) per layer, so generating a
        large batch in one go is tens to hundreds of GB -- 1000 sequences at
        max_codons=2044 needs ~400 GB and OOMs a 141 GB H200.  Chunking must not
        change the result or the ordering."""
        m = tiny()
        rng = random.Random(11)
        prots = ["M" + "".join(rng.choice(AAS) for _ in range(rng.randint(3, 40)))
                 for _ in range(25)]
        outs = constrained_sample(m, prots, device="cpu", batch_size=4)
        self.assertEqual(len(outs), len(prots))
        for out, p in zip(outs, prots):
            self.assertEqual(translate(out[:-1]), p)   # order preserved
        u = sample(m, n=13, max_codons=20, device="cpu", batch_size=5)
        self.assertEqual(len(u), 13)

    def test_ambiguous_symbols(self):
        m = tiny()
        out = constrained_sample(m, ["MXBZJ"], device="cpu", add_stop=False)[0]
        self.assertEqual(len(out), 5)
        self.assertIn(out[2], AA_TO_CODONS["D"] + AA_TO_CODONS["N"])
        self.assertIn(out[3], AA_TO_CODONS["E"] + AA_TO_CODONS["Q"])


class TestUnconstrained(unittest.TestCase):
    def test_sample_is_well_formed(self):
        m = tiny(block_size=64)
        outs = sample(m, n=6, max_codons=40, device="cpu", temperature=1.0)
        for o in outs:
            for c in o:
                self.assertEqual(len(c), 3)
                self.assertNotIn(c, ("[PAD]", "[UNK]", "[BOS]", "[EOS]"))
            self.assertEqual(len("".join(o)) % 3, 0)


class TestValidate(unittest.TestCase):
    def test_valid_and_each_failure_mode(self):
        good = ["AUG", "GCU", "AAA", "UAA"]
        r = validate_cds(good, target_protein="MAK")
        self.assertTrue(r["valid_cds"])
        self.assertTrue(r["protein_match"])
        self.assertFalse(r["internal_stop"])

        self.assertFalse(validate_cds(["GUG", "GCU", "UAA"])["starts_atg"])
        self.assertFalse(validate_cds(["AUG", "GCU"])["ends_stop"])
        self.assertTrue(validate_cds(["AUG", "UAA", "GCU", "UGA"])["internal_stop"])
        self.assertFalse(validate_cds(["AUG", "UAA", "GCU", "UGA"])["valid_cds"])
        self.assertFalse(validate_cds(good, target_protein="MAR")["protein_match"])

    def test_gc(self):
        r = validate_cds(["GGG", "CCC"])
        self.assertAlmostEqual(r["gc_content"], 1.0)
        self.assertAlmostEqual(r["gc3"], 1.0)
        self.assertAlmostEqual(validate_cds(["AAA", "UUU"])["gc_content"], 0.0)

    def test_markdown_table_matches_qc_stats_columns(self):
        rows, md = validate_batch([(["AUG", "GCU", "UAA"], "MA")], "x")
        for col in ("starts with ATG", "ends with a stop codon",
                    "has an internal stop codon", "all three satisfied"):
            self.assertIn(col, md)
        self.assertIn("target protein exact match", md)


if __name__ == "__main__":
    unittest.main()
