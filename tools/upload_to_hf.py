#!/usr/bin/env python
"""Export checkpoints and publish them to the Hugging Face Hub.

Reads a manifest of (checkpoint, repo name, description), exports each one to a
weights-only directory with ``export_for_hf.py``, writes a model card, and
uploads the result.

Authentication is taken from the ambient Hugging Face credentials: either the
``HF_TOKEN`` environment variable or the token stored by ``hf auth login``. The
token must belong to the account that owns ``--namespace`` -- a Hugging Face
*user* namespace cannot be written to by any other user -- and must carry write
permission.

Dry run first; it exports locally and prints what would be uploaded:

    python tools/upload_to_hf.py --namespace ZYMScott --out-root /path/to/staging

Then publish:

    python tools/upload_to_hf.py --namespace ZYMScott --out-root /path/to/staging --push

Each repository is created private by default. Review it on the Hub, then flip
it to public there, or pass --public to create it public from the start.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXPORT = os.path.join(HERE, "export_for_hf.py")

# (repo name, checkpoint path, released, one-line description)
#
# ``released`` marks the current release set.  An entry with ``released=False``
# is exported and card-written like any other, but only when named explicitly
# with --only or swept in with --all, so that widening the release is a
# deliberate act rather than a side effect of re-running this script.
#
# Checkpoint paths come from MRNA_GPT_RUNS so this file carries no
# machine-specific location.
RUNS = os.environ.get("MRNA_GPT_RUNS", os.path.join(ROOT, "runs"))

MANIFEST = [
    # --- pretrained backbones ---------------------------------------------
    ("mRNA-GPT-bacteria", f"{RUNS}/bacteria/model_best.pt", True,
     "Pretrained on bacterial coding sequences."),
    ("mRNA-GPT-archaea", f"{RUNS}/archaea/model_best.pt", True,
     "Pretrained on archaeal coding sequences."),
    ("mRNA-GPT-eukaryote", f"{RUNS}/eukaryote/model_best.pt", True,
     "Pretrained on eukaryotic coding sequences."),
    # --- property fine-tunes ----------------------------------------------
    ("mRNA-GPT-bacterial-expression", f"{RUNS}/bacexp_sft/model_best.pt", True,
     "Fine-tuned from mRNA-GPT-bacteria on the high-expression arm of a "
     "bacterial protein-expression library."),
    ("mRNA-GPT-fungal-expression", f"{RUNS}/fungal_sft/model_best.pt", True,
     "Fine-tuned on the high-expression arm of a fungal expression dataset."),
    ("mRNA-GPT-stability", f"{RUNS}/stability_sft/model_best.pt", True,
     "Fine-tuned on the high-stability arm of an mRNA stability dataset."),
    # --- held back from the current release --------------------------------
    ("mRNA-GPT-translation-efficiency", f"{RUNS}/te_sft/model_best.pt", False,
     "Fine-tuned on the high-translation-efficiency arm of a TE dataset."),
]


CARD = """---
license: apache-2.0
library_name: pytorch
tags:
  - biology
  - rna
  - mrna
  - codon
  - coding-sequence
  - protein-design
---

# {name}

{description}

mRNA-GPT is a decoder-only transformer language model over **codons**. The
tokenizer operates on codons rather than nucleotides, so one token is exactly
one residue position. That makes protein-constrained decoding exact: at step
*t* the logits are masked to the synonymous codon set of residue *t*, so the
generated coding sequence translates back to the requested protein **by
construction**, not with high probability.

| | |
|---|---|
| Architecture | decoder-only transformer, {n_layer} layers, {n_embd}-dim, {n_head} heads |
| Parameters | {n_params:,} |
| Vocabulary | {vocab_size} tokens (4 special + 64 codons) |
| Context | {block_size} codons |
| Positional encoding | {pos_encoding} |
| Precision | bfloat16 |
| Code | https://github.com/ZHymLumine/mRNA-GPT |

## Installation

```bash
git clone https://github.com/ZHymLumine/mRNA-GPT && cd mRNA-GPT
pip install -r requirements.txt
pip install huggingface_hub safetensors
```

```bash
huggingface-cli download {repo_id} --local-dir {name}
```

## Design a coding sequence for your protein

This is the main use. Put your target protein in a FASTA file:

```
>my_target
MKAIFVLKGSLDRDLEHHHHHHGSMSTAVLENPGLGRKLSDFGQETSYIEDNSNQ
```

```bash
python -m mrnagpt.generate \\
    --ckpt {name} \\
    --proteins my_target.fasta \\
    --temperature 0.8 --top-p 0.95 \\
    --out designs.fasta --report designs.md
```

The output FASTA keeps your sequence names, and the report states what fraction
of designs start with ATG, end with a single in-frame stop, contain no internal
stop, and translate to exactly the requested protein:

```
| | starts with ATG | ends with a stop codon | has an internal stop codon | all three satisfied |
|---|---:|---:|---:|---:|
| constrained | 100.00% | 100.00% | 0.00% | 100.00% |

- target protein exact match: 100.00% (1/1)
```

Sampling several designs per target and ranking them is the usual workflow:
repeat the call, or pass a FASTA with the target repeated.

## From Python

```python
from mrnagpt.generate import load_model, constrained_sample
from mrnagpt.vocab import translate

model = load_model("{name}", device="cuda")   # or "cpu"

target = "MKAIFVLKGSLDRDLEHHHHHHGSMSTAVLENPGLGRKLSDFGQETSYIEDNSNQ"
designs = constrained_sample(
    model, [target] * 8,          # eight independent designs
    temperature=0.8, top_p=0.95, device="cuda",
)

for codons in designs:
    cds = "".join(codons)
    assert translate(codons, stop_at_first_stop=True) == target
    print(cds)
```

`load_model` accepts this directory, the `model.safetensors` inside it, or a
`.pt` checkpoint written during training.

## Unconstrained generation

Sampling coding sequences without a target protein, for characterising what the
model has learned about the domain:

```bash
python -m mrnagpt.generate --ckpt {name} --n 100 --out sampled.fasta
```

## Files

| file | contents |
|---|---|
| `model.safetensors` | weights, bfloat16 |
| `config.json` | architecture, vocabulary size, token ids |
| `vocab.txt` | the {vocab_size}-token codon vocabulary, in id order |
| `provenance.json` | source checkpoint, training step, SHA-256 of the weights |

The tied `lm_head.weight` is not stored separately, because safetensors will not
serialise two names backed by the same storage. `config.json` records
`tie_weights`, and the key is restored when the model is built.

## Related models

{related}

## Limitations

- The model generates **coding sequences only** — no UTRs, no poly(A), no
  cap-proximal structure. Expression depends on those too.
- Sampling is per-sequence and has no notion of a host's codon supply beyond
  what it learned from the pretraining domain. For a host far from that domain,
  fine-tune rather than relying on the pretrained model.
- A property fine-tune shifts the codon distribution toward the high-property
  arm of its training set. That is a statistical shift, not a guarantee about
  any individual design: validate experimentally.
- Protein-constrained decoding guarantees the translated protein, not that the
  resulting mRNA folds or expresses well.

## License

Apache-2.0.
"""

def run_export(ckpt: str, out_dir: str) -> dict:
    import json
    cmd = [sys.executable, EXPORT, "--ckpt", ckpt, "--out", out_dir,
           "--dtype", "bf16"]
    subprocess.run(cmd, check=True)
    with open(os.path.join(out_dir, "config.json")) as fh:
        return json.load(fh)


def related_models(namespace: str, exclude: str) -> str:
    """Markdown list of the other models in the release set.

    Each card links to its siblings, so someone who lands on the fungal model
    can find the bacterial one without going back to the repository.
    """
    rows = [f"- [`{namespace}/{n}`](https://huggingface.co/{namespace}/{n}) — "
            f"{d[0].lower() + d[1:]}"
            for n, _ck, released, d in MANIFEST
            if released and n != exclude]
    return "\n".join(rows) if rows else "_None published yet._"


def write_card(out_dir: str, *, name: str, repo_id: str, description: str,
               cfg: dict, namespace: str) -> None:
    card = CARD.format(
        name=name, repo_id=repo_id, description=description,
        n_params=cfg["n_parameters"], vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"], pos_encoding=cfg["pos_encoding"],
        n_layer=cfg["n_layer"], n_embd=cfg["n_embd"], n_head=cfg["n_head"],
        related=related_models(namespace, name),
    )
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write(card)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--namespace", required=True,
                    help="Hugging Face user or organization that will own the repos")
    ap.add_argument("--out-root", required=True,
                    help="local staging directory for the exported models")
    ap.add_argument("--push", action="store_true",
                    help="actually upload; without it this is a local dry run")
    ap.add_argument("--public", action="store_true",
                    help="create the repos public (default: private)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="these model names instead of the release set")
    ap.add_argument("--all", action="store_true",
                    help="include models held back from the release set")
    args = ap.parse_args()

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        who = api.whoami()
        if who.get("name") != args.namespace and args.namespace not in [
                o.get("name") for o in who.get("orgs", [])]:
            sys.exit(f"the active token belongs to {who.get('name')!r}, which "
                     f"cannot write to the {args.namespace!r} namespace. Log in "
                     f"with a write token for that account.")

    if args.only is not None:
        todo = [m for m in MANIFEST if m[0] in args.only]
        unknown = set(args.only) - {m[0] for m in MANIFEST}
        if unknown:
            sys.exit(f"unknown model name(s): {', '.join(sorted(unknown))}")
    elif args.all:
        todo = list(MANIFEST)
    else:
        todo = [m for m in MANIFEST if m[2]]
        held = [m[0] for m in MANIFEST if not m[2]]
        if held:
            print(f"release set: {len(todo)} models; "
                  f"held back: {', '.join(held)}\n")

    published, skipped = [], []

    for name, ckpt, _released, description in todo:
        if not os.path.exists(ckpt):
            print(f"[skip] {name}: no checkpoint at {ckpt}")
            skipped.append(name)
            continue
        repo_id = f"{args.namespace}/{name}"
        out_dir = os.path.join(args.out_root, name)
        print(f"\n=== {name} ===")
        cfg = run_export(ckpt, out_dir)
        write_card(out_dir, name=name, repo_id=repo_id,
                   description=description, cfg=cfg,
                   namespace=args.namespace)

        if args.push:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id, repo_type="model",
                            private=not args.public, exist_ok=True)
            api.upload_folder(folder_path=out_dir, repo_id=repo_id,
                              repo_type="model",
                              commit_message=f"Add {name}")
            print(f"  pushed -> https://huggingface.co/{repo_id}")
        else:
            print(f"  staged -> {out_dir} (dry run; pass --push to upload)")
        published.append(name)

    print(f"\n{len(published)} prepared, {len(skipped)} skipped")
    if skipped:
        print("skipped (checkpoint not found): " + ", ".join(skipped))
    if not args.push:
        print("\nThis was a dry run. Re-run with --push to upload.")


if __name__ == "__main__":
    main()
