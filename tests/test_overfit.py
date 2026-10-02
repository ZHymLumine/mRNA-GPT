import unittest

import torch
import torch.nn.functional as F

from tests.common import *  # noqa: F401,F403
from mrnagpt.model import GPT, GPTConfig
from mrnagpt.vocab import PAD_ID


class TestOverfit(unittest.TestCase):
    def test_overfits_one_batch(self):
        """Highest-value test in the suite: catches label misalignment, a wrong
        ignore_index, a detached graph, or an LR that never gets applied."""
        torch.manual_seed(0)
        m = GPT(GPTConfig(pos_encoding="rope", n_layer=2, n_head=2, n_embd=64,
                          block_size=64, dropout=0.0))
        g = torch.Generator().manual_seed(1)
        lengths = [12, 20, 31, 8]
        x = torch.zeros(4, 40, dtype=torch.long)
        y = torch.zeros(4, 40, dtype=torch.long)
        for i, n in enumerate(lengths):
            ids = torch.randint(4, 68, (n,), generator=g)
            ids[0], ids[-1] = 2, 3
            x[i, :n - 1] = ids[:-1]
            y[i, :n - 1] = ids[1:]
        n_real = int((y != PAD_ID).sum())

        # Every row starts with [BOS], so position 0 offers one context and four
        # different targets.  That residual is irreducible; the rest must go to 0.
        import math
        first_targets = [int(y[i, 0]) for i in range(len(lengths))]
        self.assertEqual(len(set(first_targets)), len(lengths))
        floor = len(lengths) * math.log(len(lengths)) / n_real

        opt, _ = m.configure_optimizers(0.0, 3e-3, (0.9, 0.95), "cpu")
        first = last = None
        for step in range(400):
            logits, nll = m(x, y)
            loss = nll / n_real
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if step == 0:
                first = float(loss)
            last = float(loss)
        self.assertGreater(first, 3.5, "initial loss should be near ln(68)=4.22")
        self.assertLess(last, floor + 0.01,
                        f"failed to reach the irreducible floor {floor:.5f}: "
                        f"{first:.3f} -> {last:.5f}")

    def test_init_loss_is_uniform(self):
        """With PAD excluded, a fresh model must score ~ln(68)=4.22.

        The published logs show step-0 losses of 2.40-2.92, below uniform, which is
        impossible for a uniform output -- it is the scaled-residual init acting as
        a weak copy prior and scoring brilliantly on the long PAD runs.  Seeing 2.6
        here would mean PAD is back in the loss.
        """
        import math
        torch.manual_seed(0)
        m = GPT(GPTConfig(n_layer=4, n_head=4, n_embd=128, block_size=128)).eval()
        x = torch.randint(4, 68, (8, 100))
        y = torch.roll(x, -1, dims=1)
        y[:, -1] = PAD_ID
        with torch.no_grad():
            _, nll = m(x, y)
        loss = float(nll) / int((y != PAD_ID).sum())
        self.assertAlmostEqual(loss, math.log(68), delta=0.15)


if __name__ == "__main__":
    unittest.main()
