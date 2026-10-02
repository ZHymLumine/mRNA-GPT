"""Score CDS sequences with the published mRNA-LM model, fine-tuned once (LoRA)
on its own bundled protein_expression_5class task -- never fine-tuned on our
fungal data. Every CDS is scored under the SAME fixed UTR context (yeast ADH1,
see sft/data/adh1_utr_context.py), so differences between generated batches are
attributable to the CDS alone.

This module only does inference; it must be run in an environment that has
mRNA-LM's own dependencies installed, with MRNA_LM_WEIGHTS pointing at the
weight directory (`<MRNA_LM_REPO>/weights/final`) and the mRNA-LM repo itself
on sys.path.  mRNA-LM is not vendored here: check it out separately and point
MRNA_LM_REPO at it, or place it at `<MRNA_GPT_EXTERNAL>/mRNA-LM`.

Output per sequence: the full 5-class probability vector (Human-Protein-Atlas-
style expression-specificity classes used by the published task -- these are
NOT a clean low->high expression ordinal, so we report the distribution rather
than collapse it to one score) plus argmax class id.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("MRNA_GPT_ROOT")
            or Path(__file__).resolve().parents[1])
EXTERNAL = os.environ.get("MRNA_GPT_EXTERNAL") or str(ROOT / "external")
MRNA_LM_REPO = os.environ.get("MRNA_LM_REPO") or os.path.join(EXTERNAL, "mRNA-LM")
if MRNA_LM_REPO not in sys.path:
    sys.path.insert(0, MRNA_LM_REPO)

CLASS_WEIGHTS = [0.97326057, 0.48056585, 1.24829396, 1.44412955, 2.51197183]


def load_scorer(checkpoint_path: str, device: str = "cuda", num_labels: int = 5):
    """num_labels=5 loads the expression-specificity classifier; num_labels=1 the
    translation-rate regressor (the oracle Li et al. use as their primary CDS
    readout). The head shape must match the checkpoint or load_state_dict fails."""
    from FullModel import FullModel

    model = FullModel(
        num_labels=num_labels,
        class_weights=CLASS_WEIGHTS if num_labels > 1 else [],
        lorar=32,
        lalpha=32,
        ldropout=0.5,
        head_dim=768,
        head_droupout=0.5,
        useCLIP=False,
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def mytok(seq: str, kmer_len: int, s: int) -> list[str]:
    seq = seq.upper().replace("T", "U")
    return [seq[j : j + kmer_len] for j in range(0, len(seq) - kmer_len + 1, s)]


@torch.no_grad()
def score_batch(
    model,
    cds_list: list[str],
    utr5: str,
    utr3: str,
    device: str = "cuda",
    batch_size: int = 16,
) -> list[dict]:
    """Score a list of raw (unspaced) CDS nucleotide strings under one fixed UTR pair."""
    u5_tok = " ".join(mytok(utr5, 1, 1))
    u3_tok = " ".join(mytok(utr3, 1, 1))

    results = []
    for i in range(0, len(cds_list), batch_size):
        chunk = cds_list[i : i + batch_size]
        cds_tok = [" ".join(mytok(c, 3, 3)) for c in chunk]

        enc5 = model.tokenizer_5utr([u5_tok] * len(chunk), truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        enc_cds = model.tokenizer_cds(cds_tok, truncation=True, padding="max_length", max_length=1024, return_tensors="pt")
        enc3 = model.tokenizer_3utr([u3_tok] * len(chunk), truncation=True, padding="max_length", max_length=1024, return_tensors="pt")

        dummy_labels = torch.zeros(len(chunk), dtype=torch.long, device=device)
        _, logits = model(
            input_ids1=enc5["input_ids"].to(device),
            attention_mask1=enc5["attention_mask"].to(device),
            input_ids2=enc_cds["input_ids"].to(device),
            attention_mask2=enc_cds["attention_mask"].to(device),
            input_ids3=enc3["input_ids"].to(device),
            attention_mask3=enc3["attention_mask"].to(device),
            labels=dummy_labels,
        )
        logits = logits.view(len(chunk), -1)
        if logits.shape[1] == 1:                      # regression head
            for v in logits.float().cpu().numpy().flatten():
                results.append({"score": float(v)})
        else:
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            for p in probs:
                results.append(
                    {
                        "probs": p.tolist(),
                        "argmax_class": int(np.argmax(p)),
                    }
                )
    return results


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fasta", required=True, help="FASTA of CDS sequences to score")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sft.data.adh1_utr_context import ADH1_5UTR, ADH1_3UTR

    ids, seqs = [], []
    cur_id, cur_seq = None, []
    with open(args.fasta) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs.append("".join(cur_seq))
                cur_id = line[1:]
                ids.append(cur_id)
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            seqs.append("".join(cur_seq))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_scorer(args.checkpoint, device=device)
    results = score_batch(model, seqs, ADH1_5UTR, ADH1_3UTR, device=device, batch_size=args.batch_size)

    with open(args.out, "w") as fh:
        for seq_id, r in zip(ids, results):
            fh.write(json.dumps({"id": seq_id, **r}) + "\n")
    print(f"scored {len(results)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
