"""Paper figures from metrics.jsonl.

Figure C settles the PAD-dilution question on its own: the PAD-inclusive and
per-real-token curves on the same axes, annotated with the padding fraction.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

COLORS = {"archaea": "#B5651D", "bacteria": "#1F6FB4", "eukaryote": "#2E8B57",
          "smoke": "#777777"}


def read(run_dir):
    recs = []
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return recs
    with open(path) as fh:
        for line in fh:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return recs


def series(recs, event, x, y):
    pts = [(r[x], r[y]) for r in recs
           if r.get("event") == event and x in r and y in r]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def plateau_step(xs, ys, window=10, thresh=-0.02, need=3):
    """First eval where the OLS slope of val loss on log10(tokens) over a trailing
    window exceeds `thresh` nats/decade, sustained `need` times.  This -- not the
    early-stopping event -- is the quantitative answer to 'did perplexity plateau'.
    """
    import numpy as np
    hits = 0
    for i in range(window, len(xs)):
        lx = np.log10(np.maximum(np.array(xs[i - window:i], dtype=float), 1.0))
        ly = np.array(ys[i - window:i], dtype=float)
        if lx.ptp() == 0:
            continue
        slope = np.polyfit(lx, ly, 1)[0]
        hits = hits + 1 if slope > thresh else 0
        if hits >= need:
            return xs[i], slope
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories containing metrics.jsonl")
    ap.add_argument("--out", default="figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = {os.path.basename(os.path.normpath(r)): read(r) for r in args.runs
            if os.path.isdir(r)}
    data = {k: v for k, v in data.items() if v}
    if not data:
        raise SystemExit("no metrics.jsonl found in the given run dirs")

    def save(fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(args.out, f"{name}.{ext}"), dpi=200,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"  {args.out}/{name}.pdf")

    # A: validation loss (PAD-excluded)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    summary = {}
    for name, recs in data.items():
        xs, ys = series(recs, "eval", "tokens_seen", "val_loss")
        if not xs:
            continue
        c = COLORS.get(name, None)
        ax.plot(xs, ys, label=name, color=c, lw=1.6)
        i = min(range(len(ys)), key=lambda k: ys[k])
        ax.plot([xs[i]], [ys[i]], "o", color=c, ms=5)
        ps, slope = plateau_step(xs, ys)
        summary[name] = {"best_val_loss": ys[i], "best_tokens": xs[i],
                         "plateau_tokens": ps, "plateau_slope": slope,
                         "final_val_loss": ys[-1],
                         "final_ppl": math.exp(min(ys[-1], 20))}
    ax.set_xscale("log")
    ax.set_xlabel("training tokens seen (real, PAD excluded)")
    ax.set_ylabel("validation loss (nats / real token)")
    ax.set_title("Validation loss on the homology-clean split")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "figA_val_loss")

    # B: perplexity per real token
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, recs in data.items():
        xs, ys = series(recs, "eval", "tokens_seen", "val_ppl")
        if xs:
            ax.plot(xs, ys, label=name, color=COLORS.get(name), lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("training tokens seen")
    ax.set_ylabel("perplexity per real codon")
    ax.set_title("Per-codon perplexity (PAD excluded)")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "figB_perplexity")

    # C: the two conventions side by side -- the PAD-dilution comparison
    fig, axes = plt.subplots(1, len(data), figsize=(5.0 * len(data), 4.0),
                             squeeze=False)
    for ax, (name, recs) in zip(axes[0], data.items()):
        xs, ys = series(recs, "eval", "tokens_seen", "val_loss")
        xp, yp = series(recs, "eval", "tokens_seen", "val_loss_incl_pad")
        ax.plot(xs, ys, label="per real token", color="#B5651D", lw=1.6)
        ax.plot(xp, yp, label="including PAD (published convention)",
                color="#888888", lw=1.6, ls="--")
        pf = [r["pad_frac"] for r in recs if "pad_frac" in r]
        if pf and ys and yp:
            ax.annotate(f"pad fraction {100*sum(pf)/len(pf):.1f}%\n"
                        f"ppl {math.exp(min(ys[-1],20)):.2f} vs "
                        f"{math.exp(min(yp[-1],20)):.2f}",
                        xy=(0.5, 0.75), xycoords="axes fraction", fontsize=9)
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_xlabel("tokens seen")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("loss (nats / token)")
    save(fig, "figC_pad_convention")

    # D: throughput diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, recs in data.items():
        xs, ys = series(recs, "train", "global_step", "mfu_padded")
        if xs:
            axes[0].plot(xs, [100 * v for v in ys], label=name,
                         color=COLORS.get(name), lw=1.0)
        xs, ys = series(recs, "train", "global_step", "grad_norm")
        if xs:
            axes[1].plot(xs, ys, label=name, color=COLORS.get(name), lw=1.0)
    axes[0].set_ylabel("MFU (%)")
    axes[0].set_xlabel("step")
    axes[1].set_ylabel("grad norm")
    axes[1].set_xlabel("step")
    axes[1].set_yscale("log")
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    save(fig, "figD_throughput")

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
