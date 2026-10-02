# mRNA-GPT

A decoder-only transformer language model over **codons**, pretrained on coding
sequences from three domains of life (archaea, bacteria, eukaryotes) and
fine-tuned to design coding sequences with desired properties.

Because the tokenizer operates on codons rather than nucleotides, one token is
exactly one residue position. Constrained decoding therefore masks the logits at
step *t* to the synonymous codon set of residue *t*, so the translated output
equals the target protein **by construction** rather than with high probability.

- 24 layers, 1024-dim, 16 heads, 303M parameters
- Vocabulary: 68 tokens (4 special + 64 codons)
- Context: 2048 codons
- License: Apache-2.0

---

## Loading earlier checkpoints

An earlier version of this codebase used a 69-token vocabulary carrying
`[CLS]`/`[SEP]`/`[MASK]`, learned positional embeddings and a 1024-codon
context. Those checkpoints cannot be loaded with the current tokenizer.

Both vocabularies order codons alphabetically — exactly
`itertools.product("ACGU", repeat=3)` — so each codon keeps its relative
position and shifts by the one removed special token:
`codon_id_current = codon_id_old - 1`. The data-side conversion is a lookup
table in [`mrnagpt/vocab.py`](mrnagpt/vocab.py):

```python
from mrnagpt.vocab import remap_entry
ids_new = remap_entry(raw_legacy_entry)
```

The remap converts stored **data, not weights**: an older checkpoint indexes its
embedding table by the old ids and has to be fed old ids.
[`tools/eval_legacy_ckpt.py`](tools/eval_legacy_ckpt.py) loads one under the
current code, reading its original `model_args` without remapping.

If you build data for an older checkpoint with `BertTokenizerFast`, pin
`transformers==4.46.3` and `tokenizers==0.20.3`: under 5.x the same vocabulary
file is read as 5 tokens and every codon silently becomes `[UNK]`, with no error
raised. The current codebase never constructs a tokenizer object, so it is not
exposed to this.

---

## Install

```bash
conda create -n mrnagpt python=3.10 && conda activate mrnagpt
pip install -r requirements.txt
```

### Optional external tools

Nothing below is bundled. Each is resolved from `$PATH` or an environment
variable, and is needed only for the evaluation or baseline that uses it.

| Tool | Resolved via | Needed for |
|---|---|---|
| MMseqs2 | `MMSEQS_BIN` or `$PATH` | corpus clustering, novelty, homology audits |
| ViennaRNA `RNAfold` | `RNAFOLD_BIN` or `$PATH` | minimum free energy |
| R + [iCodon](https://github.com/santiago1234/iCodon) | R library path | `sft/baseline_icodon.R`, tAI weights |
| [LinearDesign](https://github.com/LinearDesignSoftware/LinearDesign) | `LINEARDESIGN_DIR` | `sft/baseline_lineardesign.py`, human codon-usage table |
| GEMORNA | `GEMORNA_DIR` | `sft/baseline_gemorna.py` |
| CodonGPT | `CODONGPT_DIR` | `sft/baseline_codongpt.py` |
| mRNA-LM | `MRNA_LM_REPO` | translation-rate / expression oracle |

Unset tool paths fall back to `$MRNA_GPT_EXTERNAL/<name>`. `$MRNA_GPT_DATA`,
`$MRNA_GPT_RUNS` and `$MRNA_GPT_EXTERNAL` default to `data/`, `runs/` and
`external/` under the repository root.

## Quickstart

Unconstrained generation:

```bash
python -m mrnagpt.generate --ckpt path/to/model_best.pt --n 16 --out out.fasta
```

Design a coding sequence for a given protein:

```bash
python -m mrnagpt.generate \
    --ckpt path/to/model_best.pt \
    --proteins targets.fasta \
    --temperature 0.8 --top-p 0.95 \
    --out designs.fasta --report designs.md
```

`--proteins` takes a FASTA of amino-acid sequences. Output is checked with
`validate_cds`: correct start codon, a single in-frame stop at the end, length
divisible by three, and translation identical to the requested protein.

## Training

Paths in `configs/*.yaml` are relative to the repository root. Override on the
command line rather than editing the file:

```bash
python -m mrnagpt.train --config configs/bacteria.yaml \
    --override data_dir=/your/data/bacteria out_dir=/your/runs/bacteria
```

Hyperparameters in the shipped configs are exactly those of the reported runs.
`configs/base.yaml` holds the shared defaults. `pbs/` contains the cluster job
scripts; they read `MRNA_GPT_ROOT`, `MRNA_GPT_DATA` and `MRNA_GPT_RUNS` from the
environment and need a scheduler allocation group set for your site.

Fine-tuning starts from a pretrained checkpoint via `init_ckpt`, which loads
model weights only — optimizer state, epoch counters and `best_val_loss` all
start over.

## Rebuilding the corpus

`scripts/` runs the pipeline end to end: CDS extraction, exact deduplication
within a domain, MMseqs2 clustering, cluster-level splitting, LMDB packing, and
a leakage report.

The split uses `mmseqs cluster --min-seq-id 0.5 -c 0.8 --cluster-mode 1`
(connected components) with whole clusters assigned to one side. Do **not**
substitute `linclust`: it links only sequences sharing exact k-mers with the
cluster representative, and leaves a large fraction of homologous pairs split
across the boundary. `scripts/06_leakage_report.py` quantifies the difference.

## Repository layout

```
mrnagpt/      model, training loop, data pipeline, vocabulary, sampling
configs/      YAML configs for pretraining, fine-tuning and ablations
scripts/      corpus construction (00–09)
sft/          property fine-tuning, baselines, evaluation panels
evaluate/     CAI/tAI, MFE, diversity, novelty, expression predictors
figures/      figure generation
tools/        checkpoint evaluation, curve plotting, sanity checks
tests/        unit tests — run with `python tests/run_all.py`
pbs/          cluster job scripts
```

## Citation

```bibtex
@article{mrnagpt,
  title   = {Large generative mRNA language foundation model for efficient
             coding sequence generation and design with mRNA-GPT},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {TODO}
}
```

## License

Apache-2.0 — see [LICENSE](LICENSE). Third-party baseline
tools invoked by `sft/baseline_*.py` are not distributed here and carry their
own licenses.
