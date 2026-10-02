"""Pretraining entry point.  Launch with torchrun.

Key differences from the published train.py, each fixing a defect that changed
the reported numbers:

* gradient accumulation consumes a **distinct** micro-batch per inner step
  (the old loop replayed the same ``X, Y``, so half the compute was wasted and
  the effective batch was half what the logs claimed);
* PAD is excluded from the loss (``ignore_index=PAD_ID``); the old run trained
  the model to emit padding on ~86% of its tokens;
* bf16 autocast with no GradScaler (the old ``GradScaler(enabled=dtype=='float32')``
  was inverted, and the training loop's unparameterised autocast silently ran fp16);
* ``--compile`` is a real flag, not the Python builtin;
* all ranks evaluate; checkpoints are atomic, verified and pruned.
"""
from __future__ import annotations

import math
import os
import signal
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrnagpt import checkpoint as ckpt_io           # noqa: E402
from mrnagpt.config import load_config, dump_config  # noqa: E402
from mrnagpt.data import (BUCKET_EDGES, CodonLMDBDataset, build_plan,  # noqa: E402
                          make_loader)
from mrnagpt.distributed import DDPInfo             # noqa: E402
from mrnagpt.evaluate import evaluate               # noqa: E402
from mrnagpt.logging_utils import MetricWriter, write_env_snapshot, git_sha  # noqa: E402
from mrnagpt.model import GPT                       # noqa: E402
from mrnagpt.vocab import PAD_ID                    # noqa: E402

_STOP = {"now": False}


def _install_signal_handlers():
    def handler(signum, frame):
        print(f"[signal] caught {signum}; will checkpoint and exit", flush=True)
        _STOP["now"] = True
    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


def lr_at(tokens_seen: int, cfg, total_tokens: int, warmup_tokens: float) -> float:
    if tokens_seen < warmup_tokens:
        return cfg.lr * tokens_seen / max(warmup_tokens, 1.0)
    if cfg.schedule == "wsd":
        decay_start = total_tokens * (1.0 - cfg.wsd_decay_frac)
        if tokens_seen < decay_start:
            return cfg.lr
        r = (tokens_seen - decay_start) / max(total_tokens - decay_start, 1.0)
        return cfg.lr + (cfg.min_lr - cfg.lr) * min(r, 1.0)
    if tokens_seen >= total_tokens:
        return cfg.min_lr
    r = (tokens_seen - warmup_tokens) / max(total_tokens - warmup_tokens, 1.0)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * r)) * (cfg.lr - cfg.min_lr)


def _eval_workers(n: int) -> int:
    """Worker count for the evaluation loaders.

    Half the training loader's workers is plenty, but ``num_workers=0`` must be
    honoured rather than floored at 2: with 0 the training loader opens the LMDB
    in the parent process, and a forked evaluation worker inheriting that open
    environment cannot reopen the same file -- lmdb refuses a second open of one
    file within a process.
    """
    return max(2, n // 2) if n else 0


def main(argv=None):
    tcfg, mcfg, resolved = load_config(argv)
    ddp = DDPInfo().init(tcfg.backend)
    device = ddp.device
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    _install_signal_handlers()

    torch.manual_seed(tcfg.seed + ddp.rank)
    np.random.seed(tcfg.seed + ddp.rank)
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch._dynamo.config.cache_size_limit = 64
    # keep the compiled kernels on shared storage: warmup is ~6 min and a job
    # that resubmits for walltime would otherwise pay it every time
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                          os.path.join(os.path.dirname(tcfg.out_dir.rstrip("/")),
                                       "_inductor_cache"))

    os.makedirs(tcfg.out_dir, exist_ok=True)
    log = MetricWriter(tcfg.out_dir, enabled=ddp.is_master)
    if ddp.is_master:
        dump_config(resolved, os.path.join(tcfg.out_dir, "config.yaml"))
        write_env_snapshot(os.path.join(tcfg.out_dir, "env.txt"))

    # ---------------- data ----------------
    train_ds = CodonLMDBDataset(os.path.join(tcfg.data_dir, tcfg.train_lmdb))
    val_ds = CodonLMDBDataset(os.path.join(tcfg.data_dir, tcfg.val_lmdb))

    # A frozen slice of *train*, scored in eval mode with the identical code path.
    # The running train loss is dropout-on, single-pass and LR-dependent, so
    # comparing it against val is not a clean generalization measurement.
    probe_ds = CodonLMDBDataset(os.path.join(tcfg.data_dir, tcfg.train_lmdb))

    plan0 = build_plan(train_ds.lengths, tcfg.token_budget, ddp.world_size,
                       tcfg.grad_accum, tcfg.seed, 0)
    tokens_per_epoch = plan0.real_tokens
    total_tokens = tokens_per_epoch * tcfg.max_epochs
    warmup_tokens = tcfg.warmup_steps * float(plan0.step_tokens.mean())

    if ddp.is_master:
        print(f"[data] {len(train_ds):,} train seqs | {plan0.n_steps:,} steps/epoch | "
              f"{tokens_per_epoch/1e9:.3f}B real tokens/epoch | "
              f"efficiency {plan0.efficiency()*100:.1f}% | "
              f"{len({(int(b), int(t)) for b, t in zip(plan0.mb_size, plan0.mb_T)})} static shapes",
              flush=True)

    # ---------------- model ----------------
    model = GPT(mcfg).to(device)
    n_params = model.get_num_params()
    optimizer, groups = model.configure_optimizers(
        tcfg.weight_decay, tcfg.lr, (tcfg.beta1, tcfg.beta2), device_type)
    if ddp.is_master:
        print(f"[model] {n_params/1e6:.2f}M non-embedding params | pos={mcfg.pos_encoding} | "
              f"decay {groups[0]} tensors/{groups[1]/1e6:.1f}M, "
              f"no-decay {groups[2]}/{groups[3]/1e3:.1f}K", flush=True)

    start_epoch, start_step, global_step, tokens_seen = 0, 0, 0, 0
    best_val, evals_since_improve = float("inf"), 0
    last_path = os.path.join(tcfg.out_dir, "ckpt_last.pt")
    if os.path.exists(last_path):
        state = ckpt_io.load(last_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, start_step = state["epoch"], state["step_in_epoch"]
        global_step, tokens_seen = state["global_step"], state["tokens_seen"]
        best_val = state.get("best_val_loss", float("inf"))
        ckpt_io.restore_rng(state.get("rng"))
        if ddp.is_master:
            print(f"[resume] epoch {start_epoch} step {start_step} "
                  f"(global {global_step}, {tokens_seen/1e9:.3f}B tokens)", flush=True)
        del state
    elif tcfg.init_ckpt:
        state = ckpt_io.load(tcfg.init_ckpt, map_location=device)
        pretrained_args = state["model_args"]
        arch_fields = ("vocab_size", "n_layer", "n_head", "n_embd", "pos_encoding")
        mismatch = {k: (pretrained_args.get(k), getattr(mcfg, k)) for k in arch_fields
                   if pretrained_args.get(k) != getattr(mcfg, k)}
        if mismatch:
            raise ValueError(f"init_ckpt architecture does not match this config: {mismatch}")
        model.load_state_dict(state["model"])
        if ddp.is_master:
            print(f"[init] loaded pretrained weights from {tcfg.init_ckpt} "
                  f"(fine-tuning from step {state.get('global_step')}); "
                  f"optimizer and epoch/step counters start fresh", flush=True)
        del state

    raw_model = model
    if tcfg.compile:
        # dynamic=False: the plan emits a fixed, known set of (B, T) shapes, so we
        # want one static graph each rather than dynamo guessing at dynamic dims
        model = torch.compile(model, dynamic=False)
    if ddp.enabled:
        model = DDP(model, device_ids=[ddp.local_rank],
                    gradient_as_bucket_view=True, broadcast_buffers=False)

    autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                      "float32": torch.float32}[tcfg.dtype]
    amp = (torch.autocast("cuda", dtype=autocast_dtype) if device_type == "cuda"
           else nullcontext())

    def _dynamo_frames() -> int:
        try:
            return int(torch._dynamo.utils.counters["frames"]["ok"])
        except Exception:
            return 0

    if tcfg.compile:
        shapes = sorted({(max(1, tcfg.token_budget // int(T)), int(T))
                         for T in BUCKET_EDGES})
        if ddp.is_master:
            print(f"[warmup] compiling {len(shapes)} bucket shapes on every rank",
                  flush=True)
        t_w = time.time()
        model.train()
        for B, T in shapes:
            xw = torch.randint(4, mcfg.vocab_size, (B, T), device=device)
            # must match the training loop's call signature exactly: model(x)
            # without targets, loss computed outside.  model(x, y) compiles a
            # different graph and leaves the real one cold.
            with amp:
                logits_w, _ = model(xw)
            F.cross_entropy(logits_w.reshape(-1, mcfg.vocab_size).float(),
                            xw.reshape(-1), reduction="sum").backward()
            model.zero_grad(set_to_none=True)
            del logits_w
        # eval mode is a *different* graph whenever dropout > 0, so the first
        # periodic eval would otherwise recompile all 17 shapes mid-run -- on 8
        # GPUs that is a compile-skew stall in the middle of training
        model.eval()
        with torch.no_grad():
            for B, T in shapes:
                xw = torch.randint(4, mcfg.vocab_size, (B, T), device=device)
                with amp:
                    model(xw)
        model.train()
        if ddp.enabled:
            ddp.barrier()
        optimizer.zero_grad(set_to_none=True)
        if device_type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        warm_frames = _dynamo_frames()
        if ddp.is_master:
            print(f"[warmup] done in {time.time()-t_w:.0f}s "
                  f"({len(shapes)} shapes x train+eval, "
                  f"{warm_frames} dynamo frames)", flush=True)
    else:
        warm_frames = 0

    t_start = time.time()
    gn_history: list[float] = []
    skipped = 0
    consec_skips = 0
    stop_all = False
    res_p = {"loss": float("nan")}

    def do_probe(step_in_epoch: int, epoch: int, val_loss: float):
        """L_train_heldout - L_val: the real generalization number."""
        res = evaluate(model, probe_ds, ddp=ddp, device=device,
                       token_budget=tcfg.token_budget, seed=777,
                       max_seqs=tcfg.train_probe_seqs,
                       num_workers=_eval_workers(tcfg.num_workers),
                       autocast_dtype=autocast_dtype)
        log.log("eval_train_probe", epoch=epoch, step_in_epoch=step_in_epoch,
                global_step=global_step, tokens_seen=tokens_seen,
                train_probe_loss=res["loss"],
                generalization_gap=val_loss - res["loss"],
                split=tcfg.train_lmdb, note=f"{res['seqs']} frozen train seqs")
        if ddp.is_master:
            print(f"[probe] train-heldout {res['loss']:.4f} | "
                  f"gap (val - train) {val_loss - res['loss']:+.4f}", flush=True)
        model.train()

    def do_eval(tag: str, step_in_epoch: int, epoch: int, ds=None, max_seqs=None):
        nonlocal best_val, evals_since_improve
        res = evaluate(model, ds or val_ds, ddp=ddp, device=device,
                       token_budget=tcfg.token_budget,
                       max_seqs=tcfg.eval_seqs if max_seqs is None else max_seqs,
                       num_workers=_eval_workers(tcfg.num_workers),
                       autocast_dtype=autocast_dtype)
        improved = res["loss"] < best_val - tcfg.early_stop_min_delta
        if tag == "eval":
            if improved:
                best_val, evals_since_improve = res["loss"], 0
            else:
                best_val = min(best_val, res["loss"])
                evals_since_improve += 1
        log.log(tag, epoch=epoch, step_in_epoch=step_in_epoch, global_step=global_step,
                tokens_seen=tokens_seen, val_loss=res["loss"], val_ppl=res["ppl"],
                val_loss_incl_pad=res["loss_incl_pad"],
                val_ppl_incl_pad=res["ppl_incl_pad"],
                val_loss_pad_positions=res.get("loss_pad_positions"),
                best_val_loss=best_val, evals_since_improve=evals_since_improve,
                split=tcfg.val_lmdb, note=f"{res['seqs']} seqs / {res['tokens']} tokens")
        if ddp.is_master:
            # NOTE on the incl-PAD figure: this model never trains on PAD
            # (ignore_index), so it scores PAD positions terribly and the
            # combined number comes out *higher*, not lower.  The published runs
            # trained on PAD and predicted it almost perfectly, which is what
            # diluted their loss down to ~0.70.  The two are therefore not
            # comparable, and the PAD-dilution correction has to be measured on
            # the published checkpoints themselves (tools/eval_legacy_ckpt.py).
            print(f"[{tag}] step {global_step} val_loss {res['loss']:.4f} "
                  f"ppl {res['ppl']:.3f} | PAD-position loss "
                  f"{res.get('loss_pad_positions', float('nan')):.3f} "
                  f"(diagnostic only) | best {best_val:.4f} "
                  f"stale {evals_since_improve}", flush=True)
        return res, improved

    def save_ckpt(kind: str, epoch: int, step_in_epoch: int):
        if not ddp.is_master:
            return
        name = {"last": "ckpt_last.pt", "best": "ckpt_best.pt"}.get(
            kind, f"ckpt_step{global_step:08d}.pt")
        info = ckpt_io.save(
            os.path.join(tcfg.out_dir, name), raw_model=raw_model, optimizer=optimizer,
            model_args=mcfg.to_dict(), train_cfg=tcfg.to_dict(), epoch=epoch,
            step_in_epoch=step_in_epoch, global_step=global_step,
            tokens_seen=tokens_seen, best_val=best_val,
            wall_seconds=time.time() - t_start, git_sha=git_sha(),
            weights_only_path=(os.path.join(tcfg.out_dir, "model_best.pt")
                               if kind == "best" else None))
        removed = ckpt_io.prune(tcfg.out_dir, tcfg.keep_last) if kind == "rolling" else []
        log.log("checkpoint", epoch=epoch, global_step=global_step, path=info["path"],
                note=f"kind={kind} size={info['size_bytes']} "
                     f"t={info['save_time_s']}s pruned={removed}")

    # ---------------- epochs ----------------
    for epoch in range(start_epoch, tcfg.max_epochs):
        plan = build_plan(train_ds.lengths, tcfg.token_budget, ddp.world_size,
                          tcfg.grad_accum, tcfg.seed, epoch)
        s0 = start_step if epoch == start_epoch else 0
        if s0 >= plan.n_steps:                 # resumed exactly at an epoch end
            start_step = 0
            continue
        step = s0 - 1
        loader = make_loader(train_ds, plan, ddp.rank, s0, tcfg.num_workers)
        it = iter(loader)
        model.train()

        # nll_real, n_real, nll_pad, n_pad, padded_tok, n_micro
        stats = torch.zeros(6, dtype=torch.float32, device=device)
        gn_sum = clip_hits = 0.0
        interval_steps = 0
        t_interval = time.time()
        data_wait = 0.0
        flops_interval = 0.0

        for step in range(s0, plan.n_steps):
            step_tok = int(plan.step_tokens[step])
            # skipping here would leave the loader iterator out of step with the
            # plan; the plan never appends grad_accum whole filler groups, so a
            # zero-token step is unreachable
            assert step_tok > 0, f"zero-token step {step} in epoch {epoch}"
            lr = lr_at(tokens_seen, tcfg, total_tokens, warmup_tokens)
            for g in optimizer.param_groups:
                g["lr"] = lr
            scale = ddp.world_size / float(step_tok)

            for j in range(tcfg.grad_accum):
                t_w = time.time()
                batch = next(it)
                data_wait += time.time() - t_w
                # the plan never yields an empty micro-batch: a rank that skipped
                # its backward would desync DDP on the following allreduce
                assert batch is not None, "empty micro-batch in train plan"
                x, y, n_real = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if ddp.enabled and global_step < 50:
                    # Structural guarantee, checked for real on the first steps:
                    # every rank must run the same (B, T) in the same micro-step.
                    # If it ever drifted, every step would wait on whichever rank
                    # drew the widest bucket, and unwarmed shapes could compile at
                    # different times on different ranks.  Costs two allreduces,
                    # only during the opening steps.
                    shp = torch.tensor([x.size(0), x.size(1)], device=device,
                                       dtype=torch.int64)
                    lo = shp.clone(); hi = shp.clone()
                    torch.distributed.all_reduce(lo, op=torch.distributed.ReduceOp.MIN)
                    torch.distributed.all_reduce(hi, op=torch.distributed.ReduceOp.MAX)
                    assert torch.equal(lo, hi), (
                        f"bucket shape desync at step {global_step} micro {j}: "
                        f"min {lo.tolist()} max {hi.tolist()}")
                sync = j == tcfg.grad_accum - 1
                ctx = nullcontext() if (sync or not ddp.enabled) else model.no_sync()
                with ctx:
                    with amp:
                        logits, _ = model(x)
                    tgt = y.reshape(-1)
                    per_tok = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)).float(), tgt,
                        reduction="none")
                    mask = tgt != PAD_ID
                    nll_real = (per_tok * mask).sum()
                    (nll_real * scale).backward()
                with torch.no_grad():
                    stats += torch.stack([
                        nll_real.detach().float(), mask.sum().float(),
                        (per_tok * (~mask)).sum().detach().float(), (~mask).sum().float(),
                        torch.tensor(float(x.numel()), device=device),
                        torch.tensor(1.0, device=device)])
                flops_interval += raw_model.flops_per_microbatch(x.size(0), x.size(1))

            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            gnf = float(gn)
            # Record every observed norm, accepted or not.  Recording only accepted
            # steps lets the rule latch: once the true distribution rises above the
            # threshold the median can never update, and every subsequent step is
            # skipped forever while the job keeps burning GPU.  That happened to
            # bacteria at step 35,725 (228 consecutive skips before it was caught).
            if math.isfinite(gnf):
                gn_history.append(gnf)
                if len(gn_history) > 200:
                    del gn_history[:-200]
            bad = not math.isfinite(gnf)
            if not bad and tcfg.grad_norm_skip_factor > 0 and len(gn_history) >= 50:
                med = float(np.median(gn_history[-100:]))
                bad = gnf > tcfg.grad_norm_skip_factor * max(med, 1e-8)
                # a relative-outlier test must never fire indefinitely
                if bad and consec_skips >= tcfg.max_consecutive_skips:
                    bad = False
                    if ddp.is_master:
                        print(f"[warn] {consec_skips} consecutive skips at step "
                              f"{global_step}; accepting to avoid a stalled run",
                              flush=True)
            if bad:
                skipped += 1
                consec_skips += 1
                optimizer.zero_grad(set_to_none=True)
                if ddp.is_master and consec_skips <= 5:
                    print(f"[warn] skipping step {global_step}: grad_norm={gnf}", flush=True)
            else:
                consec_skips = 0
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            gn_sum += 0.0 if bad else gnf
            clip_hits += float(gnf > tcfg.grad_clip)
            tokens_seen += step_tok
            global_step += 1
            interval_steps += 1

            if interval_steps >= tcfg.log_interval:
                ddp.all_reduce(stats)
                nr, cr, npd, cpd, ptok, nmb = (float(v) for v in stats)
                dt = time.time() - t_interval
                loss = nr / max(cr, 1.0)
                loss_all = (nr + npd) / max(cr + cpd, 1.0)
                flops = flops_interval * ddp.world_size
                log.log("train", epoch=epoch, step_in_epoch=step, global_step=global_step,
                        tokens_seen=tokens_seen,
                        tokens_seen_padded=int(ptok), lr=lr,
                        train_loss=loss, train_ppl=math.exp(min(loss, 20)),
                        train_loss_incl_pad=loss_all,
                        train_ppl_incl_pad=math.exp(min(loss_all, 20)),
                        grad_norm=gn_sum / max(interval_steps, 1),
                        grad_clipped_frac=clip_hits / max(interval_steps, 1),
                        skipped_steps=skipped,
                        step_time_ms=1000 * dt / interval_steps,
                        data_wait_ms=1000 * data_wait / interval_steps,
                        tokens_per_s_real=cr / dt, tokens_per_s_padded=ptok / dt,
                        pad_frac=1.0 - cr / max(ptok, 1.0),
                        mfu_padded=flops / dt / (mcfg.peak_tflops * ddp.world_size),
                        mfu_effective=(flops / dt / (mcfg.peak_tflops * ddp.world_size)
                                       * (cr / max(ptok, 1.0))),
                        gpu_mem_alloc_gb=(torch.cuda.max_memory_allocated() / 2**30
                                          if device_type == "cuda" else 0),
                        gpu_mem_reserved_gb=(torch.cuda.max_memory_reserved() / 2**30
                                             if device_type == "cuda" else 0),
                        micro_batches=int(nmb),
                        note=(f"dynamo_frames={_dynamo_frames()}"
                              if tcfg.compile else None))
                if tcfg.compile and _dynamo_frames() > warm_frames:
                    extra = _dynamo_frames() - warm_frames
                    warm_frames = _dynamo_frames()
                    if ddp.is_master:
                        print(f"[warn] {extra} new dynamo compilation(s) at step "
                              f"{global_step} -- an unwarmed shape reached the loop",
                              flush=True)
                if ddp.is_master:
                    print(f"e{epoch} s{global_step} loss {loss:.4f} ppl {math.exp(min(loss,20)):.3f} "
                          f"lr {lr:.2e} gn {gn_sum/max(interval_steps,1):.2f} "
                          f"{1000*dt/interval_steps:.0f}ms "
                          f"{cr/dt/1e3:.0f}k tok/s pad {100*(1-cr/max(ptok,1)):.1f}%",
                          flush=True)
                stats.zero_()
                gn_sum = clip_hits = 0.0
                interval_steps = 0
                data_wait = 0.0
                flops_interval = 0.0
                if device_type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                t_interval = time.time()

            if global_step % tcfg.eval_interval == 0:
                t_pause = time.time()
                res_p, improved = do_eval("eval", step + 1, epoch)
                if improved:
                    save_ckpt("best", epoch, step + 1)
                save_ckpt("last", epoch, step + 1)
                model.train()
                # eval and checkpoint time must not be charged to step_time_ms/MFU
                t_interval += time.time() - t_eval_start if False else 0
                t_interval = time.time() - (time.time() - t_interval_mark)                     if False else t_interval

                # eval + checkpoint time must not be charged to step_time_ms/MFU
                t_interval += time.time() - t_pause

                burn_in = tokens_seen < total_tokens * tcfg.early_stop_burn_in_frac
                may_stop = (epoch + 1 > tcfg.min_epochs) and not burn_in
                if may_stop and evals_since_improve >= tcfg.early_stop_patience:
                    if ddp.is_master:
                        print(f"[early-stop] no improvement in "
                              f"{evals_since_improve} evals", flush=True)
                    stop_all = ddp.broadcast_flag(True)
                elif ddp.enabled:
                    stop_all = ddp.broadcast_flag(False)

                elapsed_h = (time.time() - t_start) / 3600.0
                if elapsed_h > tcfg.max_hours:
                    if ddp.is_master:
                        print(f"[walltime] {elapsed_h:.2f}h > {tcfg.max_hours}h; "
                              f"checkpointing and exiting for resubmit", flush=True)
                    _STOP["now"] = True

            if _STOP["now"] or stop_all or (tcfg.max_steps and global_step >= tcfg.max_steps):
                break

        del it, loader
        if not (_STOP["now"] or stop_all or (tcfg.max_steps and global_step >= tcfg.max_steps)):
            res_e, _ = do_eval("eval_epoch", plan.n_steps, epoch, ds=val_ds,
                               max_seqs=0)
            do_probe(plan.n_steps, epoch, res_e["loss"])
            save_ckpt("rolling", epoch + 1, 0)
            save_ckpt("last", epoch + 1, 0)
        if _STOP["now"] or stop_all or (tcfg.max_steps and global_step >= tcfg.max_steps):
            do_probe(step + 1, epoch, res_p["loss"])
            save_ckpt("last", epoch, step + 1)
            break
        start_step = 0

    # RESUBMIT only when we stopped for walltime/signal with budget left;
    # early stopping and a completed epoch budget both mean DONE.
    interrupted = _STOP["now"] and not stop_all
    if ddp.is_master:
        marker = "RESUBMIT" if interrupted else "DONE"
        for stale in ("DONE", "RESUBMIT"):
            if stale != marker and os.path.exists(os.path.join(tcfg.out_dir, stale)):
                os.remove(os.path.join(tcfg.out_dir, stale))
        with open(os.path.join(tcfg.out_dir, marker), "w") as fh:
            fh.write(f"global_step={global_step}\ntokens_seen={tokens_seen}\n"
                     f"best_val_loss={best_val}\n")
        print(f"[exit] {marker} at step {global_step}, "
              f"{tokens_seen/1e9:.3f}B tokens, best_val {best_val:.4f}", flush=True)
    log.close()
    ddp.shutdown()


if __name__ == "__main__":
    main()
