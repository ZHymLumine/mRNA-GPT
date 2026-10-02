"""Score a trained checkpoint on a full validation set (single- or multi-GPU).

Kept out of the training loop on purpose: a full pass over the eukaryote
validation split is 9.18M sequences / 3.76B tokens, ~39 min on 8 GPUs, which has
no business running every eval_interval.  Run it once on the final checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrnagpt.data import CodonLMDBDataset                    # noqa: E402
from mrnagpt.distributed import DDPInfo                      # noqa: E402
from mrnagpt.evaluate import evaluate                        # noqa: E402
from mrnagpt.model import GPT, GPTConfig                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lmdb", required=True, action="append",
                    help="validation LMDB; repeat to score several (e.g. the "
                         "cluster-split val and the random-split control)")
    ap.add_argument("--label", action="append", default=None)
    ap.add_argument("--token-budget", type=int, default=32768)
    ap.add_argument("--max-seqs", type=int, default=0, help="0 = full pass")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    ddp = DDPInfo().init()
    device = ddp.device
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = GPTConfig(**state["model_args"])
    model = GPT(cfg).to(device)
    model.load_state_dict({(k[10:] if k.startswith("_orig_mod.") else k): v
                           for k, v in state["model"].items()})
    model.eval()

    labels = args.label or [os.path.basename(p) for p in args.lmdb]
    rows = []
    for path, label in zip(args.lmdb, labels):
        ds = CodonLMDBDataset(path)
        res = evaluate(model, ds, ddp=ddp, device=device,
                       token_budget=args.token_budget, max_seqs=args.max_seqs)
        res["label"] = label
        res["lmdb"] = path
        rows.append(res)
        if ddp.is_master:
            print(f"{label:28s} loss {res['loss']:.4f}  ppl {res['ppl']:.3f}  "
                  f"(incl-PAD {res['loss_incl_pad']:.4f}/"
                  f"{res['ppl_incl_pad']:.3f})  {res['seqs']:,} seqs", flush=True)

    if ddp.is_master:
        md = [f"# checkpoint evaluation — `{args.ckpt}`", "",
              f"- step {state.get('global_step')}, "
              f"{state.get('tokens_seen', 0)/1e9:.3f}B tokens seen",
              f"- git sha `{state.get('git_sha')}`", "",
              "| val set | loss (excl PAD) | ppl | loss (incl PAD) | "
              "ppl (incl PAD) | seqs |",
              "|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            md.append(f"| {r['label']} | {r['loss']:.4f} | {r['ppl']:.3f} | "
                      f"{r['loss_incl_pad']:.4f} | {r['ppl_incl_pad']:.3f} | "
                      f"{r['seqs']:,} |")
        if len(rows) == 2:
            d = rows[0]["loss"] - rows[1]["loss"]
            md += ["", f"**ΔL = {d:+.4f} nats** ({rows[0]['label']} − {rows[1]['label']})"]
        text = "\n".join(md)
        print("\n" + text)
        if args.report:
            with open(args.report, "w") as fh:
                fh.write(text + "\n")
            json.dump(rows, open(args.report.replace(".md", ".json"), "w"), indent=2)
    ddp.shutdown()


if __name__ == "__main__":
    main()
