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

# (repo name, checkpoint path, lineage, released, one-line description)
#
# ``released`` marks the current release set.  Entries with ``released=False``
# are exported and card-written like any other, but only when named explicitly
# with --only or swept in with --all, so that widening the release is a
# deliberate act rather than a side effect of re-running this script.
#
# Checkpoint paths are read from the environment so this file carries no
# machine-specific location: set MRNA_GPT_RUNS for the current-lineage runs and
# MRNA_GPT_LEGACY_ROOT for the preprint-era ones.
RUNS = os.environ.get("MRNA_GPT_RUNS", os.path.join(ROOT, "runs"))
LEGACY = os.environ.get("MRNA_GPT_LEGACY_ROOT", "")

MANIFEST = [
    # --- current lineage: pretrained backbones -----------------------------
    ("mRNA-GPT-bacteria", f"{RUNS}/bacteria/model_best.pt", "current", True,
     "Pretrained on bacterial coding sequences."),
    ("mRNA-GPT-archaea", f"{RUNS}/archaea/model_best.pt", "current", True,
     "Pretrained on archaeal coding sequences."),
    ("mRNA-GPT-eukaryote", f"{RUNS}/eukaryote/model_best.pt", "current", True,
     "Pretrained on eukaryotic coding sequences."),
    # --- current lineage: property fine-tunes ------------------------------
    ("mRNA-GPT-bacterial-expression", f"{RUNS}/bacexp_sft/model_best.pt",
     "current", True,
     "Fine-tuned from mRNA-GPT-bacteria on the high-expression arm of a "
     "bacterial protein-expression library."),
    ("mRNA-GPT-fungal-expression", f"{RUNS}/fungal_sft/model_best.pt",
     "current", True,
     "Fine-tuned on the high-expression arm of a fungal expression dataset."),
    ("mRNA-GPT-stability", f"{RUNS}/stability_sft/model_best.pt", "current", True,
     "Fine-tuned on the high-stability arm of an mRNA stability dataset."),
    # --- held back from the current release --------------------------------
    ("mRNA-GPT-translation-efficiency", f"{RUNS}/te_sft/model_best.pt",
     "current", False,
     "Fine-tuned on the high-translation-efficiency arm of a TE dataset."),
    # Preprint-lineage weights, for reproducing the preprint rather than for
    # new work.  Held back until the paper is through review.
    ("mRNA-GPT-bacteria-preprint", f"{LEGACY}/result/ckpt_563000.pt",
     "preprint", False,
     "Preprint-lineage bacterial model, step 563,000."),
    ("mRNA-GPT-archaea-preprint", f"{LEGACY}/result_archea/ckpt_62000.pt",
     "preprint", False,
     "Preprint-lineage archaeal model, step 62,000."),
    ("mRNA-GPT-eukaryote-preprint", f"{LEGACY}/result_ekuaryote/ckpt_694000.pt",
     "preprint", False,
     "Preprint-lineage eukaryotic model, step 694,000."),
]

CARD = """---
license: apache-2.0
library_name: pytorch
tags:
  - biology
  - rna
  - mrna
  - codon
  - dna
pipeline_tag: text-generation
---

# {name}

{description}

mRNA-GPT is a decoder-only transformer language model over **codons**. One token
is exactly one residue position, so constrained decoding can mask the logits at
step *t* to the synonymous codon set of residue *t*: the translated output
equals the target protein by construction rather than with high probability.

- 24 layers, 1024-dim, 16 heads, {n_params:,} parameters
- Vocabulary: {vocab_size} tokens
- Context: {block_size} codons
- Positional encoding: {pos_encoding}
- Code: https://github.com/ZHymLumine/mRNA-GPT

{lineage_note}## Usage

```bash
git clone https://github.com/ZHymLumine/mRNA-GPT && cd mRNA-GPT
pip install -r requirements.txt
huggingface-cli download {repo_id} --local-dir {name}
```

```bash
python -m mrnagpt.generate \\
    --ckpt {name}/model.safetensors \\
    --proteins targets.fasta \\
    --temperature 0.8 --top-p 0.95 \\
    --out designs.fasta
```

`provenance.json` in this repository records the source checkpoint, its training
step, and the SHA-256 of the weights file.

## License

Apache-2.0.
"""

PREPRINT_NOTE = """## Tokenizer compatibility

These weights use the earlier 69-token vocabulary (`[CLS]`/`[SEP]`/`[MASK]`),
learned positional embeddings and a 1024-codon context, so they cannot be loaded
with the current 68-token tokenizer. Load them with `tools/eval_legacy_ckpt.py`,
which reads their original `model_args`.

Both vocabularies order codons alphabetically, exactly
`itertools.product("ACGU", repeat=3)`, so `codon_id_current = codon_id_old - 1`.
That mapping converts stored data, not weights.

If you build data for these weights with `BertTokenizerFast`, pin
`transformers==4.46.3` and `tokenizers==0.20.3`: under 5.x the same vocabulary
file is read as 5 tokens and every codon silently becomes `[UNK]`.

"""

CURRENT_NOTE = ""


def run_export(ckpt: str, out_dir: str, lineage: str) -> dict:
    import json
    cmd = [sys.executable, EXPORT, "--ckpt", ckpt, "--out", out_dir,
           "--dtype", "bf16", "--lineage", lineage]
    subprocess.run(cmd, check=True)
    with open(os.path.join(out_dir, "config.json")) as fh:
        return json.load(fh)


def write_card(out_dir: str, *, name: str, repo_id: str, description: str,
               lineage: str, cfg: dict) -> None:
    card = CARD.format(
        name=name, repo_id=repo_id, description=description, lineage=lineage,
        n_params=cfg["n_parameters"], vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"], pos_encoding=cfg["pos_encoding"],
        lineage_note=PREPRINT_NOTE if lineage == "preprint" else CURRENT_NOTE,
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
        todo = [m for m in MANIFEST if m[3]]
        held = [m[0] for m in MANIFEST if not m[3]]
        if held:
            print(f"release set: {len(todo)} models; "
                  f"held back: {', '.join(held)}\n")

    published, skipped = [], []

    for name, ckpt, lineage, _released, description in todo:
        if not os.path.exists(ckpt):
            print(f"[skip] {name}: no checkpoint at {ckpt}")
            skipped.append(name)
            continue
        repo_id = f"{args.namespace}/{name}"
        out_dir = os.path.join(args.out_root, name)
        print(f"\n=== {name} ({lineage}) ===")
        cfg = run_export(ckpt, out_dir, lineage)
        write_card(out_dir, name=name, repo_id=repo_id,
                   description=description, lineage=lineage, cfg=cfg)

        if args.push:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id, repo_type="model",
                            private=not args.public, exist_ok=True)
            api.upload_folder(folder_path=out_dir, repo_id=repo_id,
                              repo_type="model",
                              commit_message=f"Add {name} ({lineage} lineage)")
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
