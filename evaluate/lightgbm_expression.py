"""LightGBM property predictor: a genuinely independent evaluator.

Deliberately a different model class (gradient-boosted trees on engineered
codon-usage features) from anything used to filter the SFT training set (here,
filtering is done directly on real measured Value -- no predictor at all, see
sft/prepare_sft_data.py), and trained/validated/tested on the homology-clean
split (reports/*_leakage.md), not the CSV's original split. This is what makes
it a legitimate answer to the circularity concern: it never touches the
sequences used to build the SFT training set, and its own train/val/test
partition has zero homology leakage.

`--prefix` selects the dataset: `fungal_expression` (measured fungal expression)
or `mrna_stability` (measured mRNA half-life), reading
`<data-dir>/<prefix>_{train,val,test}.csv`. Both go through the identical
featurization and fitting code, so the two predictors are comparable and there
is only one place where leakage could be reintroduced.

Every encoder/scaler is fit on TRAIN ONLY, after the split -- the leakage bug
found in the prior attempt (StandardScaler.fit on the full dataset's y-values
before splitting, in vita/rna2expression/rna2expression.py) is what this file
exists to avoid.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import lightgbm as lgb
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from evaluate.codon_metrics import GENETIC_CODE, cai, gc3_content, gc_content, tai

CODONS_60 = sorted(c for c in GENETIC_CODE if GENETIC_CODE[c] != "*")


def to_codons(seq: str) -> list[str]:
    seq = seq.upper().replace("T", "U")
    return [seq[i:i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def featurize(seq: str, cai_ref: dict) -> np.ndarray:
    codons = to_codons(seq)
    n = max(len(codons), 1)
    freq = {c: 0 for c in CODONS_60}
    for c in codons:
        if c in freq:
            freq[c] += 1
    freq_vec = [freq[c] / n for c in CODONS_60]
    extra = [len(seq), gc_content(codons), gc3_content(codons),
            cai(codons, cai_ref), tai(codons)]
    return np.array(freq_vec + extra, dtype=np.float32)


FEATURE_NAMES = CODONS_60 + ["length_nt", "gc", "gc3", "cai", "tai"]


def load_split(csv_path: str):
    rows = list(csv.DictReader(open(csv_path)))
    return [r["Sequence"] for r in rows], np.array([float(r["Value"]) for r in rows])


def build_features(seqs: list[str], cai_ref: dict) -> np.ndarray:
    return np.stack([featurize(s, cai_ref) for s in seqs])


def train(data_dir: str, out_dir: str, seed: int = 42,
          prefix: str = "fungal_expression") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    tr_seq, tr_y = load_split(os.path.join(data_dir, f"{prefix}_train.csv"))
    va_seq, va_y = load_split(os.path.join(data_dir, f"{prefix}_val.csv"))
    te_seq, te_y = load_split(os.path.join(data_dir, f"{prefix}_test.csv"))

    # CAI reference built from TRAIN ONLY (top 10% by measured Value), never val/test
    from evaluate.codon_metrics import build_cai_reference, save_cai_reference
    order = np.argsort(tr_y)[::-1]
    top_n = max(1, len(tr_seq) // 10)
    ref_seqs = [tr_seq[i] for i in order[:top_n]]
    cai_ref = build_cai_reference([to_codons(s) for s in ref_seqs])
    save_cai_reference(cai_ref, os.path.join(out_dir, "cai_reference.json"))

    Xtr, Xva, Xte = (build_features(s, cai_ref) for s in (tr_seq, va_seq, te_seq))

    model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=31,
                              min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                              random_state=seed)
    model.fit(Xtr, tr_y, eval_set=[(Xva, va_y)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

    def metrics(X, y, label):
        pred = model.predict(X, num_iteration=model.best_iteration_)
        return {
            "split": label, "n": len(y),
            "r2": float(r2_score(y, pred)),
            "pearson_r": float(pearsonr(y, pred)[0]),
            "spearman_r": float(spearmanr(y, pred)[0]),
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(mean_squared_error(y, pred) ** 0.5),
        }

    results = {"dataset": prefix,
              "train": metrics(Xtr, tr_y, "train"), "val": metrics(Xva, va_y, "val"),
              "test": metrics(Xte, te_y, "test"), "best_iteration": model.best_iteration_}

    model.booster_.save_model(os.path.join(out_dir, "lgbm_expression.txt"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    with open(os.path.join(out_dir, "feature_names.json"), "w") as fh:
        json.dump(FEATURE_NAMES, fh)
    return results


def load_predictor(out_dir: str):
    model = lgb.Booster(model_file=os.path.join(out_dir, "lgbm_expression.txt"))
    from evaluate.codon_metrics import load_cai_reference
    cai_ref = load_cai_reference(os.path.join(out_dir, "cai_reference.json"))
    return model, cai_ref


def predict(seqs: list[str], model, cai_ref) -> np.ndarray:
    X = build_features(seqs, cai_ref)
    return model.predict(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="sft/data")
    ap.add_argument("--out-dir", default="sft/lightgbm_expression")
    ap.add_argument("--prefix", default="fungal_expression",
                    help="dataset basename under --data-dir: fungal_expression | mrna_stability")
    args = ap.parse_args()
    results = train(args.data_dir, args.out_dir, prefix=args.prefix)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
