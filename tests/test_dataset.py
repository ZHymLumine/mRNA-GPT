import gzip
import unittest

import numpy as np

from tests.common import HAVE_DATA, VAL_LMDB, VAL_TXT
from mrnagpt.data import (BUCKET_EDGES, CodonLMDBDataset, PlanBatchSampler,
                          build_plan, collate)
from mrnagpt.vocab import BOS_ID, EOS_ID, PAD_ID, decode_codons


@unittest.skipUnless(HAVE_DATA, "archaea LMDB not available")
class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = CodonLMDBDataset(VAL_LMDB, check=True)
        cls.lines = []
        with gzip.open(VAL_TXT, "rt") as fh:
            for i, line in enumerate(fh):
                cls.lines.append(line.split())
                if i + 1 >= 2000:
                    break

    def test_items_match_text_and_lengths(self):
        rng = np.random.default_rng(0)
        for i in rng.choice(len(self.lines), size=300, replace=False):
            i = int(i)
            ids, T = self.ds[(i, 2048)]
            self.assertEqual(int(ids[0]), BOS_ID)
            self.assertEqual(int(ids[-1]), EOS_ID)
            self.assertEqual(len(ids), len(self.lines[i]) + 2)
            self.assertEqual(len(ids), int(self.ds.lengths[i]))
            self.assertEqual(decode_codons(ids), self.lines[i])

    def test_sentinel_row(self):
        ids, T = self.ds[(-1, 128)]
        self.assertIsNone(ids)
        self.assertEqual(T, 128)

    def test_collate_shapes_and_shift(self):
        batch = [self.ds[(0, 512)], self.ds[(1, 512)], self.ds[(-1, 512)]]
        x, y, n_real = collate(batch)
        self.assertEqual(x.shape, (3, 512))
        self.assertEqual(y.shape, (3, 512))
        ids0, _ = batch[0]
        n = len(ids0)
        # y is x shifted by one
        self.assertTrue((y[0, :n - 2] == x[0, 1:n - 1]).all())
        self.assertTrue((x[2] == PAD_ID).all())      # sentinel row is all PAD
        self.assertTrue((y[2] == PAD_ID).all())
        self.assertEqual(n_real, sum(len(b[0]) - 1 for b in batch if b[0] is not None))

    def test_bucket_width_respected(self):
        plan = build_plan(self.ds.lengths[:20000], 8192, 2, 1, 42, 0)
        sub = CodonLMDBDataset(VAL_LMDB)
        for k, mb in enumerate(PlanBatchSampler(plan, 0)):
            T = mb[0][1]
            self.assertIn(T, BUCKET_EDGES)
            for i, t in mb:
                self.assertEqual(t, T)
                if i >= 0:
                    self.assertLessEqual(int(sub.lengths[i]), T)
            if k > 40:
                break

    def test_workers_agree(self):
        """Lazily opening the LMDB is what makes num_workers>0 safe; the old code
        opened it in __init__ and shared the mmap across forks."""
        from torch.utils.data import DataLoader
        plan = build_plan(self.ds.lengths[:5000], 8192, 1, 1, 42, 0)
        outs = []
        for nw in (0, 2):
            ds = CodonLMDBDataset(VAL_LMDB)
            dl = DataLoader(ds, batch_sampler=PlanBatchSampler(plan, 0),
                            collate_fn=collate, num_workers=nw)
            outs.append([(x.sum().item(), y.sum().item(), n)
                         for x, y, n in list(dl)[:20]])
        self.assertEqual(outs[0], outs[1])


if __name__ == "__main__":
    unittest.main()
