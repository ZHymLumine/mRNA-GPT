"""Steps A-D of the single-GPU smoke test: environment, data path, memory sweep,
compile warm-up.  Each check has an expected value and a red flag; the point is
to fail here rather than on 8 H200s."""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrnagpt.data import (BUCKET_EDGES, CodonLMDBDataset, PlanBatchSampler,  # noqa: E402
                          build_plan, collate, make_loader)
from mrnagpt.model import GPT, GPTConfig                                     # noqa: E402
from mrnagpt.vocab import PAD_ID                                             # noqa: E402

OUT: list[str] = []
_LIVE = {"path": None}


def say(line=""):
    print(line, flush=True)
    OUT.append(line)
    # write incrementally: PBS spools stdout until the job ends, so the report
    # file on /groups is the only way to watch a long preflight in flight
    if _LIVE["path"]:
        with open(_LIVE["path"], "w") as fh:
            fh.write("\n".join(OUT) + "\n")


def check(name, ok, detail=""):
    say(f"{'  [ok] ' if ok else '  [!!] '}{name}{('  ' + detail) if detail else ''}")
    return ok


# --------------------------------------------------------------------------- #
def step_a():
    say("## A. environment")
    say(f"  torch {torch.__version__} cuda={torch.version.cuda}")
    if not torch.cuda.is_available():
        check("cuda available", False, "no GPU visible -- A/C/D are meaningless")
        return False
    props = torch.cuda.get_device_properties(0)
    say(f"  device: {props.name}  {props.total_memory/2**30:.1f} GiB  "
        f"sm_{props.major}{props.minor}")
    check("H200-class device", "H200" in props.name or props.major >= 9, props.name)
    check("memory ~141 GiB", props.total_memory > 130 * 2**30,
          f"{props.total_memory} B")

    # Flash SDPA: there is no flash-attn package here, and a silent fallback to
    # the math backend costs 2-3x and changes the memory model entirely.
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = torch.randn(2, 16, 512, 64, device="cuda", dtype=torch.bfloat16)
    ok = True
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    except Exception as exc:
        ok = False
        say(f"      {exc}")
    check("flash SDPA kernel usable (head_dim=64, bf16, causal)", ok)
    return ok


def step_b(data_dir):
    say()
    say("## B. data path")
    train = os.path.join(data_dir, "train_codon.lmdb")
    ds = CodonLMDBDataset(train, check=True)
    L = ds.lengths
    say(f"  {len(ds):,} sequences  mean {L.mean():.1f}  "
        f"p50 {int(np.percentile(L,50))}  p95 {int(np.percentile(L,95))}  max {L.max()}")
    check("max length fits the largest bucket", L.max() <= BUCKET_EDGES[-1])
    ids, _ = ds[(0, 2048)]
    check("entries decode with the 68-token vocab", int(ids.max()) < 68,
          f"max id {int(ids.max())}")

    plan = build_plan(L, 32768, 1, 1, 42, 0)
    eff = plan.efficiency()
    check("padding fraction <= 12%", 1 - eff <= 0.12, f"pad {100*(1-eff):.1f}%")
    say(f"  {plan.n_steps:,} micro-batches/epoch, "
        f"{len({(int(b), int(t)) for b, t in zip(plan.mb_size, plan.mb_T)})} static shapes")
    say(f"  pad-to-2048 baseline would be {100*(L-1).sum()/(len(L)*2048):.1f}% efficient")

    for nw in (2, 4, 8):
        loader = make_loader(ds, plan, 0, num_workers=nw)
        it = iter(loader)
        next(it)
        t0 = time.time()
        n_seq = n_tok = 0
        for _ in range(60):
            x, y, nr = next(it)
            n_seq += x.size(0)
            n_tok += nr
        dt = time.time() - t0
        say(f"  num_workers={nw}: {n_seq/dt:,.0f} seq/s  {n_tok/dt/1e3:,.0f}k tok/s")
        del it, loader
    say("  (need >= ~718 seq/s / 208k tok/s on one GPU to keep it fed)")
    return ds, plan


def step_c(budgets=(8192, 16384, 32768, 65536, 98304)):
    say()
    say("## C. token-budget sweep")
    if not torch.cuda.is_available():
        say("  skipped (no GPU)")
        return
    say("  | tokens/GPU | bucket T | reserved GiB | tok/s |")
    say("  |---:|---:|---:|---:|")
    for budget in budgets:
        for T in (256, 2048):
            B = max(1, budget // T)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                model = GPT(GPTConfig()).cuda()
                opt, _ = model.configure_optimizers(0.1, 3e-4, (0.9, 0.95), "cuda")
                x = torch.randint(4, 68, (B, T), device="cuda")
                y = x.clone()
                for i in range(6):
                    if i == 3:
                        torch.cuda.synchronize()
                        t0 = time.time()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, nll = model(x, y)
                    (nll / max(int((y != PAD_ID).sum()), 1)).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                dt = (time.time() - t0) / 3
                res = torch.cuda.max_memory_reserved() / 2**30
                say(f"  | {budget} | {T} | {res:.1f} | {B*T/dt/1e3:,.0f}k |")
            except torch.cuda.OutOfMemoryError:
                say(f"  | {budget} | {T} | OOM | - |")
            finally:
                del model, opt
                torch.cuda.empty_cache()
    say("  Flash makes attention memory O(T), so T=2048 and T=256 should be within")
    say("  ~5% at equal token count.  A large gap means flash is NOT being used.")


def step_d():
    say()
    say("## D. torch.compile warm-up")
    if not torch.cuda.is_available():
        say("  skipped (no GPU)")
        return
    torch._dynamo.config.cache_size_limit = 64
    model = torch.compile(GPT(GPTConfig()).cuda(), dynamic=False)
    budget = 32768
    t0 = time.time()
    for T in BUCKET_EDGES:
        B = max(1, budget // T)
        x = torch.randint(4, 68, (B, T), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, nll = model(x, x)
        nll.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    frames = torch._dynamo.utils.counters["frames"]["ok"]
    say(f"  compiled {len(BUCKET_EDGES)} bucket shapes in {time.time()-t0:.0f}s "
        f"({frames} dynamo frames)")
    say("  (frames counts forward and backward graphs, so pass 1 > len(BUCKET_EDGES)"
        " is expected; what matters is that pass 2 adds none and is fast)")
    t0 = time.time()
    for T in BUCKET_EDGES:
        B = max(1, budget // T)
        x = torch.randint(4, 68, (B, T), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, nll = model(x, x)
        nll.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    after = torch._dynamo.utils.counters["frames"]["ok"]
    dt2 = time.time() - t0
    check("no recompilation on the second pass", after == frames,
          f"{frames} -> {after}")
    # the decisive signal: a cached pass is ~2 orders of magnitude faster than one
    # that recompiles, regardless of how frames are counted
    check("second pass is cache-fast (<5s for all shapes)", dt2 < 5.0,
          f"{dt2:.1f}s")
    t0 = time.time()
    for T in BUCKET_EDGES:
        B = max(1, budget // T)
        x = torch.randint(4, 68, (B, T), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, nll = model(x, x)
        nll.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    third = torch._dynamo.utils.counters["frames"]["ok"]
    check("no recompilation on the third pass", third == after, f"{after} -> {third}")
    say(f"  pass2 {dt2:.1f}s  pass3 {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--budgets", default="8192,16384,32768,65536,98304",
                    help="comma-separated token budgets for the step-C sweep")
    args = ap.parse_args()

    _LIVE["path"] = args.out
    say(f"# preflight  {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    step_a()
    step_b(args.data_dir)
    if not args.skip_sweep:
        step_c(tuple(int(b) for b in args.budgets.split(",") if b))
    step_d()
    if args.out:
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
