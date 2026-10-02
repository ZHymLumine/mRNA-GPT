"""Run the whole suite: python tests/run_all.py  (no pytest dependency)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The login node routinely sits at load average ~20; letting torch grab every
# core makes the suite thrash rather than run faster.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import torch  # noqa: E402

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(os.path.abspath(__file__)),
                            pattern="test_*.py", top_level_dir=os.path.dirname(
                                os.path.dirname(os.path.abspath(__file__))))
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)
