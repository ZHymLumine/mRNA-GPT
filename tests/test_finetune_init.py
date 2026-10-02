"""init_ckpt: fine-tuning from a pretrained checkpoint, distinct from resuming
an interrupted run of the SAME job. Runs mrnagpt.train as a subprocess against
a tiny synthetic LMDB, since this exercises the real CLI/config path end to end."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml

from tests.common import *  # noqa: F401,F403
from mrnagpt import checkpoint as ckpt_io
from mrnagpt.data import build_lengths
from mrnagpt.model import GPT, GPTConfig

PYTHON = sys.executable


def _make_synthetic_lmdb(path, n=40, seed=0):
    import lmdb
    rng = np.random.default_rng(seed)
    env = lmdb.open(path, subdir=False, map_size=10**7)
    with env.begin(write=True) as txn:
        for i in range(n):
            ln = 8 + int(rng.integers(0, 5))
            ids = np.array([2, 3] + list(rng.integers(5, 69, size=ln)) + [3, 3],
                           dtype=np.uint8)
            txn.put(str(i).encode(), ids.tobytes())
    env.close()
    build_lengths(path)


class TestInitCkpt(unittest.TestCase):
    def test_finetune_resets_optimizer_and_step_counters(self):
        cfg = GPTConfig(n_layer=2, n_embd=32, n_head=2, block_size=64, dropout=0.0)
        m = GPT(cfg)
        opt, _ = m.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        pretrained = os.path.join(d, "pretrained.pt")
        ckpt_io.save(pretrained, raw_model=m, optimizer=opt, model_args=cfg.to_dict(),
                    train_cfg={}, epoch=5, step_in_epoch=0, global_step=9999,
                    tokens_seen=123456, best_val=1.23, wall_seconds=1.0)

        data_dir = os.path.join(d, "data")
        os.makedirs(data_dir, exist_ok=True)
        train_lmdb = os.path.join(data_dir, "train_codon.lmdb")
        _make_synthetic_lmdb(train_lmdb)
        for ext in ("", ".lengths.npy", ".lengths.npy.meta"):
            shutil.copy(train_lmdb + ext, os.path.join(data_dir, "val_codon.lmdb" + ext))

        out_dir = os.path.join(d, "sft_run")
        tiny_cfg = dict(
            domain="synthetic", data_dir=data_dir, out_dir=out_dir,
            val_lmdb="val_codon.lmdb", token_budget=1024, grad_accum=1, max_epochs=1,
            lr=1e-4, min_lr=1e-5, warmup_steps=1, eval_interval=5, eval_seqs=32,
            num_workers=0, log_interval=2, compile=False, max_steps=5,
            init_ckpt=pretrained,
            model=dict(pos_encoding="rope", block_size=64, n_layer=2, n_head=2,
                      n_embd=32, bias=False, dropout=0.0),
        )
        cfg_path = os.path.join(d, "tiny_sft.yaml")
        yaml.safe_dump(tiny_cfg, open(cfg_path, "w"))

        r = subprocess.run([PYTHON, "-m", "mrnagpt.train", "--config", cfg_path],
                           capture_output=True, text=True, cwd=os.path.dirname(
                               os.path.dirname(os.path.abspath(__file__))))
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("[init] loaded pretrained weights", r.stdout)

        new_state = ckpt_io.load(os.path.join(out_dir, "ckpt_last.pt"), map_location="cpu")
        self.assertLess(new_state["global_step"], 100,
                        "should start fresh, not resume from the pretrained ckpt's step 9999")
        self.assertEqual(new_state["epoch"], 1)  # epoch 0 completed normally (3 steps < max_steps=5)


if __name__ == "__main__":
    unittest.main()
