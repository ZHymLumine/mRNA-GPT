"""Build the ``<lmdb>.lengths.npy`` cache.  CPU-only; run on rt_HC, never in a GPU job."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrnagpt.data import build_lengths, lengths_path_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lmdb_paths", nargs="+")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for path in args.lmdb_paths:
        out = lengths_path_for(path)
        if os.path.exists(out) and not args.force:
            print(f"skip (exists): {out}")
            continue
        t0 = time.time()
        lens = build_lengths(path)
        print(f"{path}\n  -> {out}  n={lens.shape[0]:,}  "
              f"mean={lens.mean():.1f}  p50={int(lens.mean())}  max={lens.max()}  "
              f"[{time.time()-t0:.1f}s]", flush=True)


if __name__ == "__main__":
    main()
