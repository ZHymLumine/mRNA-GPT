import math
import os
import tempfile
import unittest

import torch

from tests.common import *  # noqa: F401,F403
from mrnagpt import checkpoint as ckpt_io
from mrnagpt.data import BUCKET_EDGES
from mrnagpt.model import GPT, GPTConfig, KVCache


class TestModel(unittest.TestCase):
    def test_published_param_count(self):
        """24L/1024d/16h must still be the 302.11M the published logs report."""
        m = GPT(GPTConfig(pos_encoding="rope"))
        self.assertAlmostEqual(m.get_num_params() / 1e6, 302.11, places=1)
        # 24 * (4d^2 attn + 8d^2 mlp) + wte + 2 LayerNorms per block + ln_f
        closed_form = 24 * 12 * 1024 ** 2 + 68 * 1024 + (24 * 2 + 1) * 1024
        self.assertEqual(m.get_num_params(), closed_form)
        m2 = GPT(GPTConfig(pos_encoding="learned"))
        self.assertEqual(m2.get_num_params(non_embedding=True), closed_form)
        self.assertEqual(sum(p.numel() for p in m2.parameters()),
                         closed_form + 2048 * 1024)

    def test_rope_has_no_wpe(self):
        rope = GPT(GPTConfig(pos_encoding="rope", n_layer=1, n_embd=32, n_head=2))
        learned = GPT(GPTConfig(pos_encoding="learned", n_layer=1, n_embd=32, n_head=2))
        self.assertNotIn("wpe", rope.transformer)
        self.assertIn("wpe", learned.transformer)
        self.assertFalse(any("wpe" in k for k in rope.state_dict()))
        # rope caches must not be persisted, so rope_factor can change at load time
        self.assertFalse(any("inv_freq" in k for k in rope.state_dict()))

    def test_weight_tying(self):
        m = GPT(GPTConfig(n_layer=1, n_embd=32, n_head=2))
        self.assertIs(m.transformer.wte.weight, m.lm_head.weight)

    def test_init_std(self):
        """Regression test for the published kaiming_normal_(nonlinearity='relu'),
        which left c_attn/c_fc at std 0.044, about 2x too large."""
        cfg = GPTConfig(n_layer=8, n_embd=512, n_head=8)
        m = GPT(cfg)
        fc = torch.cat([b.mlp.c_fc.weight.flatten() for b in m.transformer.h])
        proj = torch.cat([b.mlp.c_proj.weight.flatten() for b in m.transformer.h])
        self.assertAlmostEqual(float(fc.std()), 0.02, delta=0.002)
        self.assertAlmostEqual(float(proj.std()),
                               0.02 / math.sqrt(2 * cfg.n_layer), delta=0.0006)

    def test_all_bucket_shapes(self):
        m = GPT(GPTConfig(n_layer=1, n_embd=32, n_head=2, block_size=2048)).eval()
        with torch.no_grad():
            for T in BUCKET_EDGES:
                B = max(1, 4096 // T)
                x = torch.randint(4, 68, (B, T))
                logits, _ = m(x)
                self.assertEqual(tuple(logits.shape), (B, T, 68))

    def test_kv_cache_matches_full_forward(self):
        for pe in ("rope", "learned"):
            with self.subTest(pe=pe):
                m = GPT(GPTConfig(pos_encoding=pe, n_layer=2, n_embd=64, n_head=4,
                                  block_size=64)).eval()
                x = torch.randint(4, 68, (2, 12))
                with torch.no_grad():
                    full, _ = m(x)
                    cache = KVCache(m.config, 2, 64, x.device, torch.float32)
                    outs = []
                    for t in range(x.size(1)):
                        step, _ = m(x[:, t:t + 1], cache=cache, pos_offset=cache.length)
                        outs.append(step)
                    inc = torch.cat(outs, dim=1)
                torch.testing.assert_close(full, inc, atol=1e-4, rtol=1e-4)

    def test_checkpoint_roundtrip(self):
        cfg = GPTConfig(n_layer=2, n_embd=64, n_head=4, block_size=128)
        m = GPT(cfg)
        opt, _ = m.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
        x = torch.randint(4, 68, (2, 16))
        with torch.no_grad():
            before, _ = m(x)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ckpt.pt")
            ckpt_io.save(p, raw_model=m, optimizer=opt, model_args=cfg.to_dict(),
                         train_cfg={}, epoch=1, step_in_epoch=5, global_step=7,
                         tokens_seen=99, best_val=1.5, wall_seconds=1.0)
            st = ckpt_io.load(p)
            m2 = GPT(GPTConfig(**st["model_args"]))
            m2.load_state_dict(st["model"])
            with torch.no_grad():
                after, _ = m2(x)
            torch.testing.assert_close(before, after, atol=0, rtol=0)
            self.assertEqual(st["global_step"], 7)

    def test_prune_keeps_last_n(self):
        with tempfile.TemporaryDirectory() as d:
            for s in (10, 20, 30, 40, 50):
                open(os.path.join(d, f"ckpt_step{s:08d}.pt"), "w").close()
            open(os.path.join(d, "ckpt_best.pt"), "w").close()
            removed = ckpt_io.prune(d, keep_last=2)
            left = sorted(os.listdir(d))
            self.assertEqual(len(removed), 3)
            self.assertIn("ckpt_best.pt", left)
            self.assertIn("ckpt_step00000050.pt", left)
            self.assertNotIn("ckpt_step00000010.pt", left)

    def test_rope_cache_is_eager_not_lazy(self):
        """A lazily-built RoPE cache puts a Python branch on sequence length inside
        the traced region: the first shape compiled takes the build branch and the
        rest do not, so that shape recompiles when it next appears.  Measured cost
        was one extra graph per run, always at the bucket where T == block_size."""
        m = GPT(GPTConfig(pos_encoding="rope", n_layer=1, n_embd=32, n_head=2,
                          block_size=512))
        self.assertEqual(m.rope.cos_cached.size(2), 512)
        self.assertEqual(m.rope.sin_cached.size(2), 512)
        # the widest training shape must not trigger a rebuild
        before = m.rope.cos_cached.data_ptr()
        m.rope(512, 0, torch.device("cpu"), torch.float32)
        self.assertEqual(m.rope.cos_cached.data_ptr(), before)
        # decoding past block_size may extend, and must stay correct
        cos, sin = m.rope(600, 0, torch.device("cpu"), torch.float32)
        self.assertEqual(cos.size(2), 600)
        # caches stay out of the checkpoint so rope_factor can change at load time
        self.assertFalse([k for k in m.state_dict() if "cached" in k or "inv_freq" in k])

    def test_flops_uses_actual_shape(self):
        m = GPT(GPTConfig(n_layer=24, n_embd=1024, n_head=16))
        short = m.flops_per_microbatch(128, 256)
        long = m.flops_per_microbatch(16, 2048)
        # equal token counts, but the quadratic term makes the wide bucket costlier
        self.assertGreater(long, short)
        self.assertLess(long / short, 1.5)


if __name__ == "__main__":
    unittest.main()


class TestRNGRestore(unittest.TestCase):
    """Checkpoints are loaded with map_location=<cuda device>, which drags the RNG
    states onto the GPU; set_rng_state then raises "RNG state must be a
    torch.ByteTensor".  Resume had only been exercised on CPU, so this crashed the
    first real 8-GPU resume."""

    def test_restores_from_non_cpu_byte_states(self):
        st = ckpt_io.rng_states()
        # emulate what map_location does to the saved tensors
        st["torch"] = st["torch"].to(torch.int64)
        ckpt_io.restore_rng(st)                    # must not raise
        self.assertIsInstance(torch.get_rng_state(), torch.ByteTensor)

    def test_survives_a_corrupt_state(self):
        ckpt_io.restore_rng({"torch": torch.zeros(3), "numpy": None, "python": None})
        ckpt_io.restore_rng(None)

    def test_roundtrip_is_faithful(self):
        torch.manual_seed(1234)
        st = ckpt_io.rng_states()
        a = torch.randn(5)
        ckpt_io.restore_rng(st)
        torch.testing.assert_close(torch.randn(5), a)
