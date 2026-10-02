"""Tests for sft/prepare_property_split.py's pre-clustering stage.

The MMseqs2 steps need the binary and minutes of CPU, so they are exercised by
actually running the pipeline (reports/mrna_stability_leakage.md). What is
tested here is the part where a silent mistake would poison everything
downstream without failing: replicate collapsing. mRNA_Stability.csv carries
65,356 rows over 29,949 distinct sequences, and 8,928 of those sequences appear
in more than one of the CSV's own train/val/test splits. If replicates were not
collapsed before clustering, the identical sequence would land on both sides of
the split -- the exact leakage the file exists to remove.
"""
import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sft.prepare_property_split import load_and_dedup


def write_csv(rows):
    fh = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
    w = csv.DictWriter(fh, fieldnames=["Sequence", "Value", "Split"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
    fh.close()
    return fh.name


A = "AUGGCCUAA"          # 3 codons, ATG start, stop end
B = "AUGCCCGGGUAG"       # 4 codons
C = "AUGUAAGGGUUU"       # internal stop, kept but flagged
BAD_CHAR = "AUGNNNUAA"
BAD_FRAME = "AUGGCCUA"


class TestLoadAndDedup(unittest.TestCase):
    def test_replicates_collapse_to_the_mean(self):
        path = write_csv([
            {"Sequence": A, "Value": "1.0", "Split": "train"},
            {"Sequence": A, "Value": "2.0", "Split": "val"},
            {"Sequence": A, "Value": "3.0", "Split": "val"},
            {"Sequence": B, "Value": "-1.0", "Split": "test"},
        ])
        recs, stats = load_and_dedup(path, "t", 2044)
        os.unlink(path)

        self.assertEqual(stats["rows"], 4)
        self.assertEqual(stats["unique"], 2)
        self.assertEqual(stats["replicated"], 1)

        a = next(r for r in recs if r["Sequence"] == A)
        self.assertAlmostEqual(a["Value"], 2.0)
        self.assertEqual(a["n_measurements"], 3)
        self.assertAlmostEqual(a["value_sd"], 1.0)
        b = next(r for r in recs if r["Sequence"] == B)
        self.assertEqual(b["n_measurements"], 1)
        self.assertEqual(b["value_sd"], 0.0)

    def test_multi_split_sequences_are_counted_and_get_a_majority_label(self):
        path = write_csv([
            {"Sequence": A, "Value": "1.0", "Split": "train"},
            {"Sequence": A, "Value": "2.0", "Split": "val"},
            {"Sequence": A, "Value": "3.0", "Split": "val"},
            {"Sequence": B, "Value": "0.0", "Split": "test"},
        ])
        recs, stats = load_and_dedup(path, "t", 2044)
        os.unlink(path)

        # A straddles train and val in the source CSV: that IS the exact-duplicate
        # leakage, so it has to be reported, not quietly resolved.
        self.assertEqual(stats["multi_split_seqs"], 1)
        self.assertEqual(stats["multi_split_rows"], 3)
        a = next(r for r in recs if r["Sequence"] == A)
        self.assertEqual(a["split_old"], "val")            # 2 of 3 rows
        self.assertEqual(a["split_old_all"], "train|val")

    def test_ids_are_stable_and_in_input_order(self):
        path = write_csv([{"Sequence": B, "Value": "0.0", "Split": "train"},
                          {"Sequence": A, "Value": "0.0", "Split": "train"},
                          {"Sequence": B, "Value": "1.0", "Split": "train"}])
        recs, _ = load_and_dedup(path, "prop", 2044)
        os.unlink(path)
        self.assertEqual([r["seq_id"] for r in recs], ["prop_000000", "prop_000001"])
        self.assertEqual(recs[0]["Sequence"], B)           # first seen wins its slot

    def test_alphabet_and_frame_are_filtered_cds_syntax_only_recorded(self):
        path = write_csv([
            {"Sequence": A, "Value": "0.0", "Split": "train"},
            {"Sequence": C, "Value": "0.0", "Split": "train"},
            {"Sequence": BAD_CHAR, "Value": "0.0", "Split": "train"},
            {"Sequence": BAD_FRAME, "Value": "0.0", "Split": "train"},
        ])
        recs, stats = load_and_dedup(path, "t", 2044)
        os.unlink(path)

        self.assertEqual(stats["bad_char"], 1)
        self.assertEqual(stats["bad_frame"], 1)
        self.assertEqual(stats["unique"], 2)               # A and C survive
        # scripts/01_extract_cds.py records these three rather than filtering on
        # them, so the SFT corpus matches how the pretraining corpus was built
        self.assertEqual(stats["starts_aug"], 2)
        self.assertEqual(stats["internal_stop"], 1)        # C only
        self.assertEqual(stats["ends_stop"], 1)            # A only; C ends UUU

    def test_dna_input_is_converted_to_rna(self):
        path = write_csv([{"Sequence": "ATGGCCTAA", "Value": "0.0", "Split": "train"}])
        recs, stats = load_and_dedup(path, "t", 2044)
        os.unlink(path)
        self.assertEqual(stats["unique"], 1)
        self.assertEqual(recs[0]["Sequence"], A)

    def test_too_long_is_dropped_so_it_cannot_exceed_block_size(self):
        path = write_csv([{"Sequence": "AUG" * 5 + "UAA", "Value": "0.0", "Split": "train"},
                          {"Sequence": A, "Value": "0.0", "Split": "train"}])
        recs, stats = load_and_dedup(path, "t", 4)          # 4-codon cap
        os.unlink(path)
        self.assertEqual(stats["bad_length"], 1)
        self.assertEqual([r["Sequence"] for r in recs], [A])


if __name__ == "__main__":
    unittest.main()
