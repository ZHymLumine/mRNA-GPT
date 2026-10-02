import unittest

import numpy as np

from tests.common import HAVE_DATA, TRAIN_LMDB
from mrnagpt.data import BUCKET_EDGES, PlanBatchSampler, build_plan


def synth_lengths(n=60000, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.lognormal(5.6, 0.7, n).astype(np.int64), 3, 2046)


class TestPlan(unittest.TestCase):
    def setUp(self):
        self.L = synth_lengths()

    def test_deterministic(self):
        a = build_plan(self.L, 32768, 8, 2, 42, 0)
        b = build_plan(self.L, 32768, 8, 2, 42, 0)
        for f in ("order", "mb_start", "mb_size", "mb_T", "groups", "step_tokens"):
            np.testing.assert_array_equal(getattr(a, f), getattr(b, f))
        c = build_plan(self.L, 32768, 8, 2, 42, 1)
        self.assertFalse(np.array_equal(a.order, c.order), "epoch must reshuffle")

    def test_rank_step_counts_equal(self):
        for ws in (1, 2, 4, 8):
            for ga in (1, 2, 4):
                p = build_plan(self.L, 16384, ws, ga, 42, 0)
                counts = {len(PlanBatchSampler(p, r)) for r in range(ws)}
                self.assertEqual(counts, {p.n_steps * ga}, f"ws={ws} ga={ga}")

    def test_all_ranks_share_bucket_width(self):
        """Mismatched widths across ranks cost 10-30% to stragglers and risk a
        compile-skew NCCL timeout, neither visible on one GPU."""
        ws, ga = 8, 2
        p = build_plan(self.L, 32768, ws, ga, 42, 0)
        its = [iter(PlanBatchSampler(p, r)) for r in range(ws)]
        for _ in range(min(300, p.n_steps * ga)):
            widths = {b[0][1] for b in (next(i) for i in its) if b}
            self.assertLessEqual(len(widths), 1)

    def test_budget_and_shapes(self):
        p = build_plan(self.L, 32768, 8, 2, 42, 0)
        shapes = set()
        for m in range(p.mb_start.size):
            B, T = int(p.mb_size[m]), int(p.mb_T[m])
            self.assertLessEqual(B * T, 32768)
            shapes.add((B, T))
            seg = p.order[p.mb_start[m]:p.mb_start[m] + B]
            real = seg[seg >= 0]
            if real.size:
                self.assertLessEqual(int(self.L[real].max()), T)
        self.assertLessEqual(len(shapes), len(BUCKET_EDGES))

    def test_every_sequence_used_exactly_once(self):
        p = build_plan(self.L, 32768, 8, 2, 42, 0)
        real = p.order[p.order >= 0]
        self.assertEqual(len(real), len(set(real.tolist())))
        self.assertEqual(len(real), len(self.L))

    def test_val_covers_everything(self):
        p = build_plan(self.L, 32768, 4, 1, 7, 0)
        seen = []
        for r in range(4):
            for batch in PlanBatchSampler(p, r):
                seen.extend(i for i, _ in batch if i >= 0)
        self.assertEqual(sorted(seen), list(range(len(self.L))))

    def test_step_tokens_match_consumption(self):
        ws, ga = 4, 2
        p = build_plan(self.L, 16384, ws, ga, 42, 0)
        its = [iter(PlanBatchSampler(p, r)) for r in range(ws)]
        for s in range(min(50, p.n_steps)):
            tot = 0
            for _ in range(ga):
                for it in its:
                    tot += sum(int(self.L[i]) - 1 for i, _ in next(it) if i >= 0)
            self.assertEqual(tot, int(p.step_tokens[s]))

    def test_resume_is_a_suffix(self):
        p = build_plan(self.L, 16384, 2, 2, 42, 0)
        k = 13
        full = list(PlanBatchSampler(p, 1))
        tail = list(PlanBatchSampler(p, 1, start_step=k))
        self.assertEqual(full[k * p.grad_accum:], tail)

    @unittest.skipUnless(HAVE_DATA, "archaea LMDB not available")
    def test_real_data_efficiency(self):
        from mrnagpt.data import load_lengths
        L = load_lengths(TRAIN_LMDB)
        p = build_plan(L, 32768, 8, 1, 42, 0)
        self.assertGreater(p.efficiency(), 0.88)
        naive = (L - 1).sum() / (len(L) * 2048)
        self.assertLess(naive, 0.20)          # the published pad-to-block_size cost


if __name__ == "__main__":
    unittest.main()
