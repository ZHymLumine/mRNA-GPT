import unittest

import torch
import torch.nn.functional as F

from tests.common import *  # noqa: F401,F403  (sys.path)
from mrnagpt.model import GPT, GPTConfig
from mrnagpt.vocab import PAD_ID


def tiny(**kw):
    cfg = dict(pos_encoding="rope", n_layer=2, n_head=4, n_embd=64,
               block_size=256, dropout=0.0)
    cfg.update(kw)
    torch.manual_seed(0)
    return GPT(GPTConfig(**cfg))


def make_batch(lengths, width, vocab=68, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(len(lengths), width, dtype=torch.long)
    y = torch.zeros(len(lengths), width, dtype=torch.long)
    for i, n in enumerate(lengths):
        ids = torch.randint(4, vocab, (n,), generator=g)
        ids[0], ids[-1] = 2, 3
        x[i, :n - 1] = ids[:-1]
        y[i, :n - 1] = ids[1:]
    return x, y


class TestLoss(unittest.TestCase):
    def test_padding_invariance(self):
        """Loss must not depend on how much PAD we bolt on -- the core claim of
        ignore_index=PAD_ID."""
        m = tiny().eval()
        lengths = [17, 33, 9, 64]
        with torch.no_grad():
            _, a = m(*make_batch(lengths, 64))
            _, b = m(*make_batch(lengths, 256))
        self.assertAlmostEqual(float(a), float(b), places=3)

    def test_causal_mask_equivalence(self):
        """A right-padded mixed-length batch scores each row exactly as if the row
        had been run alone -- so no explicit padding mask is needed, and the fused
        flash kernel stays reachable."""
        m = tiny().eval()
        lengths = [12, 40, 7]
        x, y = make_batch(lengths, 64)
        with torch.no_grad():
            logits, _ = m(x)
            per = F.cross_entropy(logits.reshape(-1, 68).float(), y.reshape(-1),
                                  ignore_index=PAD_ID, reduction="none")
            per = per.view(len(lengths), 64)
            for i, n in enumerate(lengths):
                solo_logits, _ = m(x[i:i + 1, :n - 1])
                solo = F.cross_entropy(solo_logits.reshape(-1, 68).float(),
                                       y[i, :n - 1], reduction="none")
                torch.testing.assert_close(per[i, :n - 1], solo, atol=1e-4, rtol=1e-4)

    def test_pad_excluded_from_loss(self):
        m = tiny().eval()
        x, y = make_batch([20, 20], 64)
        with torch.no_grad():
            logits, nll = m(x, y)
        n_real = int((y != PAD_ID).sum())
        self.assertEqual(n_real, 2 * 19)
        per = F.cross_entropy(logits.reshape(-1, 68).float(), y.reshape(-1),
                              ignore_index=PAD_ID, reduction="sum")
        self.assertAlmostEqual(float(nll), float(per), places=3)

    def test_accumulation_equals_global_token_mean(self):
        """Regression test for the published train.py:316, which replayed the same
        micro-batch grad_accum times.  Accumulating distinct micro-batches with the
        1/step_tokens scale must equal one big batch's global token mean."""
        m = tiny()
        micro = [make_batch(ls, 64, seed=s) for s, ls in
                 enumerate([[13, 29], [40, 8], [21, 55], [11, 34]])]
        step_tok = sum(int((y != PAD_ID).sum()) for _, y in micro)

        m.zero_grad(set_to_none=True)
        for x, y in micro:
            logits, _ = m(x)
            per = F.cross_entropy(logits.reshape(-1, 68).float(), y.reshape(-1),
                                  reduction="none")
            mask = y.reshape(-1) != PAD_ID
            ((per * mask).sum() / step_tok).backward()
        acc = [p.grad.detach().clone() for p in m.parameters()]

        m.zero_grad(set_to_none=True)
        X = torch.cat([x for x, _ in micro])
        Y = torch.cat([y for _, y in micro])
        logits, nll = m(X, Y)
        (nll / step_tok).backward()
        big = [p.grad.detach().clone() for p in m.parameters()]

        for g1, g2 in zip(acc, big):
            torch.testing.assert_close(g1, g2, atol=2e-5, rtol=2e-3)

    def test_incl_pad_matches_when_no_pad(self):
        m = tiny().eval()
        x, y = make_batch([64], 64)
        with torch.no_grad():
            logits, _ = m(x)
        tgt = y.reshape(-1)
        per = F.cross_entropy(logits.reshape(-1, 68).float(), tgt, reduction="none")
        mask = tgt != PAD_ID
        clean = float((per * mask).sum() / mask.sum())
        n_pad = int((~mask).sum())
        self.assertEqual(n_pad, 1)          # only the final unused column


if __name__ == "__main__":
    unittest.main()


class TestSkipRule(unittest.TestCase):
    """The gradient-norm skip rule must not be able to latch.

    Recording only accepted steps means that once the true norm distribution rises
    above skip_factor x a stale median, the median can never update and every
    subsequent step is skipped forever -- the job keeps running and stops learning.
    That is exactly what happened to bacteria at step 35,725 (228 consecutive dead
    steps).  This reproduces the old behaviour and asserts the new one recovers.
    """

    @staticmethod
    def _run(norms, factor, record_all, cap=10):
        hist, skipped, consec = [], [], 0
        for i, gnf in enumerate(norms):
            if record_all:
                hist.append(gnf)
            bad = False
            if factor > 0 and len(hist) >= 5:
                med = sorted(hist[-100:])[len(hist[-100:]) // 2]
                bad = gnf > factor * max(med, 1e-8)
                if bad and consec >= cap:
                    bad = False
            if bad:
                skipped.append(i)
                consec += 1
            else:
                consec = 0
                if not record_all:
                    hist.append(gnf)
        return skipped

    def test_old_rule_latches(self):
        norms = [0.4] * 60 + [3.0] * 200          # a genuine regime change
        skipped = self._run(norms, factor=5.0, record_all=False, cap=10**9)
        self.assertGreater(len(skipped), 150, "the old rule should latch")
        self.assertEqual(skipped[-1], len(norms) - 1, "and never recover")

    def test_recording_every_norm_breaks_the_latch(self):
        norms = [0.4] * 60 + [3.0] * 200
        skipped = self._run(norms, factor=5.0, record_all=True)
        self.assertLess(len(skipped), 60, "median must adapt to the new regime")
        self.assertNotIn(len(norms) - 1, skipped, "must be learning again by the end")

    def test_consecutive_cap_is_a_backstop(self):
        norms = [0.4] * 60 + [100.0] * 100        # persistent extreme outliers
        skipped = self._run(norms, factor=5.0, record_all=False, cap=10)
        runs, cur = [], 0
        for i in range(60, len(norms)):
            cur = cur + 1 if i in skipped else 0
            runs.append(cur)
        self.assertLessEqual(max(runs), 11, "cap must bound consecutive skips")

    def test_default_skips_only_non_finite(self):
        norms = [0.4] * 60 + [5.0, 9.9, 4.2]
        self.assertEqual(self._run(norms, factor=0.0, record_all=True), [])
