"""Full independent-evaluator comparison between de novo generation from the
pretrained eukaryote checkpoint and the fungal-SFT checkpoint (before/after
fine-tuning on real measured fungal expression data).

Evaluators, each answering a distinct question:
  - CDS validity (ATG start / stop end / no internal stop)  -- already in the
    generate.py report, reproduced here for one combined table.
  - GC / GC3, CAI, tAI                                       -- codon usage.
  - MFE (ViennaRNA RNAfold)                                  -- secondary structure.
  - novelty: nearest-neighbor %identity against the homology-clean fungal
    training set (sft/data/fungal_expression_train.csv) -- is the model just
    regurgitating training sequences?
  - diversity: k-mer entropy + pairwise Jensen-Shannon divergence within each
    generated batch -- mode-collapse check.
  - LightGBM-predicted expression (sft/lightgbm_expression/) -- a model class
    never used to build the SFT training set (see evaluate/lightgbm_expression.py).
  - mRNA-LM 5-class expression-specificity score, under the fixed ADH1 UTR
    context -- run separately (needs the mRNA-LM environment), merged in here
    if its output file is present.

With --per-target, FASTA ids of the form "{label}|{target}|{i}" (written by
sft/generate_target_panel.py) are grouped per target protein, so every metric is
a paired before/after at fixed amino-acid sequence -- codon choice is then the
only thing that can move a number.

Needs lightgbm importable, plus mmseqs2 and RNAfold resolvable (see
evaluate/diversity.py's _mmseqs_bin() and evaluate/mfe.py).
"""
import csv
import gzip
import json
import random
import statistics
import sys

sys.path.insert(0, ".")

from sft.paths import RUNS

from evaluate.codon_metrics import cai, gc3_content, gc_content, load_cai_reference, load_tai_weights, tai
from evaluate.diversity import batch_diversity, nearest_neighbor_identity
from evaluate.lightgbm_expression import load_predictor, predict as lgbm_predict
from evaluate.mfe import calculate_mfe_batch


def read_fasta(path: str) -> dict[str, str]:
    seqs, cur_id, cur = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(cur)
                cur_id = line[1:].split()[0]
                cur = []
            else:
                cur.append(line)
        if cur_id is not None:
            seqs[cur_id] = "".join(cur)
    return seqs


def to_codons(seq: str) -> list[str]:
    seq = seq.upper().replace("T", "U")
    return [seq[i : i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def summarize(name: str, vals: list[float]) -> dict:
    vals = [v for v in vals if v == v]  # drop NaN
    if not vals:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": len(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def synonymous_variant_stats(seqs: list[str], max_pairs: int = 20_000,
                             seed: int = 0) -> dict:
    """Mode-collapse check for a group of sequences encoding the SAME protein.

    batch_diversity's k-mer JSD is near-blind in that setting -- amino-acid
    composition is fixed by construction, so k-mer profiles are similar however
    the codons were chosen.  What actually varies is codon choice per position,
    so measure that directly: pairwise fraction of positions carrying the same
    codon, plus the fraction of distinct sequences.
    """
    n = len(seqs)
    out = {"n_seqs": n, "unique_fraction": (len(set(seqs)) / n) if n else 0.0}
    if n < 2 or len({len(s) for s in seqs}) != 1:
        out["note"] = "unequal lengths (not one fixed protein): codon identity skipped"
        return out
    rows = [to_codons(s) for s in seqs]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        pairs = random.Random(seed).sample(pairs, max_pairs)
    vals = [sum(a == b for a, b in zip(rows[i], rows[j])) / len(rows[i])
            for i, j in pairs]
    out["n_pairs"] = len(pairs)
    out["mean_pairwise_codon_identity"] = statistics.mean(vals)
    out["median_pairwise_codon_identity"] = statistics.median(vals)
    return out


def evaluate_seqs(label: str, seqs: dict[str, str], cai_ref: dict, tai_w: dict,
                  lgbm_model, lgbm_cai_ref, reference_seqs: dict[str, str]) -> dict:
    ids = list(seqs.keys())
    raw = [seqs[i] for i in ids]
    codon_lists = [to_codons(s) for s in raw]

    gc = [gc_content(c) for c in codon_lists]
    gc3 = [gc3_content(c) for c in codon_lists]
    cai_v = [cai(c, cai_ref) for c in codon_lists]
    tai_v = [tai(c, tai_w) for c in codon_lists]

    mfe_res = calculate_mfe_batch(raw, ids=ids)
    mfe_v = [mfe_res[i]["mfe"] for i in ids]
    mfe_per_nt = [mfe_res[i]["mfe_per_nt"] for i in ids]

    nn = nearest_neighbor_identity({i: seqs[i] for i in ids}, reference_seqs)
    novelty_identity = [nn[i]["best_identity"] for i in ids]

    div = batch_diversity(raw)

    lgbm_scores = lgbm_predict(raw, lgbm_model, lgbm_cai_ref).tolist()

    return {
        "label": label,
        "n": len(ids),
        "gc": summarize("gc", gc),
        "gc3": summarize("gc3", gc3),
        "cai": summarize("cai", cai_v),
        "tai": summarize("tai", tai_v),
        "mfe": summarize("mfe", mfe_v),
        "mfe_per_nt": summarize("mfe_per_nt", mfe_per_nt),
        "novelty_best_identity_pct": summarize("novelty_best_identity_pct", novelty_identity),
        "diversity": div,
        "synonymous_variants": synonymous_variant_stats(raw),
        "lgbm_predicted_expression": summarize("lgbm_predicted_expression", lgbm_scores),
    }


def evaluate_batch(label: str, fasta_path: str, *args) -> dict:
    return evaluate_seqs(label, read_fasta(fasta_path), *args)


def group_by_target(seqs: dict[str, str]) -> dict[str, dict[str, str]]:
    """FASTA id "{label}|{target}|{i}" -> {target: {id: seq}}, order preserved."""
    groups: dict[str, dict[str, str]] = {}
    for sid, seq in seqs.items():
        parts = sid.split("|")
        if len(parts) < 3:
            raise ValueError(f"id {sid!r} is not '{{label}}|{{target}}|{{i}}': "
                             "--per-target needs sft/generate_target_panel.py output")
        groups.setdefault(parts[1], {})[sid] = seq
    return groups


def load_fungal_train_reference(csv_path: str) -> dict[str, str]:
    ref = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            ref[row["seq_id"]] = row["Sequence"]
    return ref


def metric_rows(property_name: str) -> list[tuple[str, str, int]]:
    return [
        ("cai", "CAI", 3), ("tai", "tAI", 3), ("gc", "GC", 4), ("gc3", "GC3", 4),
        ("mfe_per_nt", "MFE/nt", 4),
        ("lgbm_predicted_expression", f"LightGBM predicted {property_name}", 3),
    ]


def render_paired_md(results: dict, labels: tuple[str, str],
                     property_name: str = "expression") -> str:
    """Paired per-target table: same protein, same evaluator, two checkpoints."""
    a, b = labels
    targets = list(results[a].keys())
    md = ["# Target-protein panel: paired independent evaluation of constrained "
          "generation", "",
          f"For every target protein each checkpoint generated n="
          f"{results[a][targets[0]]['n']} synonymous variants; the amino-acid sequence "
          "is fixed, so any difference in a metric can only come from codon choice.", ""]
    for key, name, nd in metric_rows(property_name):
        md += [f"## {name}", "",
               f"| target protein | {a} | {b} | Δ |", "|---|---:|---:|---:|"]
        for t in targets:
            x = results[a][t][key].get("mean")
            y = results[b][t][key].get("mean")
            if x is None or y is None:
                md.append(f"| {t} | – | – | – |")
                continue
            md.append(f"| {t} | {x:.{nd}f} | {y:.{nd}f} | {y - x:+.{nd}f} |")
        md.append("")
    md += ["## Synonymous-variant diversity (mode-collapse check)", "",
           f"| target protein | model | unique-sequence fraction | pairwise codon identity mean |",
           "|---|---|---:|---:|"]
    for t in targets:
        for lab in (a, b):
            s = results[lab][t]["synonymous_variants"]
            ci = s.get("mean_pairwise_codon_identity")
            md.append(f"| {t} | {lab} | {100*s['unique_fraction']:.1f}% | "
                      f"{'–' if ci is None else f'{100*ci:.1f}%'} |")
    md += ["", "## Nearest-neighbour identity to the training set (memorisation check)", "",
           "No hit above the MMseqs2 threshold counts as 0; under constrained "
           "generation the protein is already fixed, so this mainly confirms that "
           "training sequences were not copied at the codon level either.", "",
           f"| target protein | {a} | {b} |", "|---|---:|---:|"]
    for t in targets:
        md.append(f"| {t} | {results[a][t]['novelty_best_identity_pct']['mean']:.2f}% "
                  f"| {results[b][t]['novelty_best_identity_pct']['mean']:.2f}% |")
    return "\n".join(md) + "\n"


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained-fasta", default=str(RUNS / "fungal_sft/generation/pretrained_eukaryote.fasta"))
    ap.add_argument("--sft-fasta", default=str(RUNS / "fungal_sft/generation/fungal_sft.fasta"))
    ap.add_argument("--lgbm-dir", default="sft/lightgbm_expression")
    ap.add_argument("--train-csv", default="sft/data/fungal_expression_train.csv")
    ap.add_argument("--out", default=str(RUNS / "fungal_sft/generation/evaluator_comparison.json"))
    ap.add_argument("--per-target", action="store_true",
                    help="group by the target field of '{label}|{target}|{i}' FASTA ids")
    ap.add_argument("--md-out", default=None, help="paired markdown summary (--per-target only)")
    ap.add_argument("--labels", default="pretrained_eukaryote,fungal_sft",
                    help="comma-separated (pretrained, fine-tuned) labels; must match the "
                         "FASTA id prefixes written by sft/generate_target_panel.py")
    ap.add_argument("--property-name", default="expression",
                    help="what the LightGBM predictor in --lgbm-dir predicts, for the "
                         "table heading (e.g. stability)")
    args = ap.parse_args()

    cai_ref = load_cai_reference(f"{args.lgbm_dir}/cai_reference.json")
    tai_w = load_tai_weights()
    lgbm_model, lgbm_cai_ref = load_predictor(args.lgbm_dir)
    reference_seqs = load_fungal_train_reference(args.train_csv)

    labels = tuple(l.strip() for l in args.labels.split(","))
    if len(labels) != 2:
        raise SystemExit(f"--labels needs exactly two names, got {labels}")
    fastas = dict(zip(labels, (args.pretrained_fasta, args.sft_fasta)))
    ev_args = (cai_ref, tai_w, lgbm_model, lgbm_cai_ref, reference_seqs)

    results: dict = {}
    for label in labels:
        if not args.per_target:
            print(f"evaluating {label} ({fastas[label]}) ...", flush=True)
            results[label] = evaluate_batch(label, fastas[label], *ev_args)
            print(json.dumps(results[label], indent=2), flush=True)
            continue
        results[label] = {}
        for target, sub in group_by_target(read_fasta(fastas[label])).items():
            print(f"evaluating {label} / {target} (n={len(sub)}) ...", flush=True)
            results[label][target] = evaluate_seqs(f"{label}|{target}", sub, *ev_args)
            print(json.dumps(results[label][target], indent=2), flush=True)

    if args.per_target:
        common = set(results[labels[0]]) & set(results[labels[1]])
        missing = (set(results[labels[0]]) | set(results[labels[1]])) - common
        if missing:
            raise SystemExit(f"targets present for only one checkpoint: {sorted(missing)} "
                             "-- a paired comparison needs both")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {args.out}")

    if args.per_target and args.md_out:
        with open(args.md_out, "w") as fh:
            fh.write(render_paired_md(results, labels, args.property_name))
        print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()
