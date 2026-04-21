"""Embedding + classifier evaluation pipeline.

Reads embedding parquet files listed in a manifest, runs 5-fold CV with
LogisticRegression and XGBoost, then saves scores to results/.

Usage (from finetune/ dir):
    python classify.py
    python classify.py --manifest results/qwen-embedding-8b-richard/manifest.json
    python classify.py --manifest exp/exp1/results/baseline/manifest.json --out-dir exp/exp1/results/baseline
"""
from __future__ import annotations

import argparse
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5

CLASSIFIERS = {
    "logreg": LogisticRegression(
        random_state=SEED,
        max_iter=2000,
        class_weight="balanced",
    ),
    "xgboost": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=4,
    ),
}


def load_frame(parquet_path: str) -> pd.DataFrame:
    frame = pd.read_parquet(parquet_path)
    frame["label"] = (frame["promise_status"] == "Yes").astype(int)
    return frame


def build_xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(frame["embedding"].tolist(), dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int64)
    return X, y


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    classifier,
    model_name: str,
    clf_name: str,
) -> dict:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    binary_scores, macro_scores, weighted_scores = [], [], []
    oof_pred = np.zeros_like(y)
    oof_proba = np.zeros(len(y), dtype=np.float32)

    for fold_id, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        fold_clf = clone(classifier)
        fold_clf.fit(X[train_idx], y[train_idx])
        pred = fold_clf.predict(X[valid_idx])
        proba = fold_clf.predict_proba(X[valid_idx])[:, 1]
        oof_pred[valid_idx] = pred
        oof_proba[valid_idx] = proba.astype(np.float32)

        b = f1_score(y[valid_idx], pred, average="binary")
        m = f1_score(y[valid_idx], pred, average="macro")
        w = f1_score(y[valid_idx], pred, average="weighted")
        binary_scores.append(b)
        macro_scores.append(m)
        weighted_scores.append(w)
        print(f"  {model_name} | {clf_name} | fold {fold_id}: binary={b:.4f} macro={m:.4f} weighted={w:.4f}")

    report = classification_report(y, oof_pred, target_names=["No", "Yes"], output_dict=True)
    cm = confusion_matrix(y, oof_pred).tolist()

    return {
        "embedding_model": model_name,
        "classifier": clf_name,
        "n_splits": N_SPLITS,
        "rows": int(len(y)),
        "embedding_dim": int(X.shape[1]),
        "positive_ratio": float(y.mean()),
        "cv_binary_f1_mean": float(np.mean(binary_scores)),
        "cv_binary_f1_std": float(np.std(binary_scores)),
        "cv_macro_f1_mean": float(np.mean(macro_scores)),
        "cv_macro_f1_std": float(np.std(macro_scores)),
        "cv_weighted_f1_mean": float(np.mean(weighted_scores)),
        "cv_weighted_f1_std": float(np.std(weighted_scores)),
        "oof_classification_report": report,
        "oof_confusion_matrix": cm,
    }


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("Summary (sorted by macro F1)")
    print("=" * 70)
    header = f"{'embedding':<22} {'classifier':<10} {'binary':>8} {'macro':>8} {'weighted':>10}"
    print(header)
    print("-" * len(header))
    sorted_results = sorted(results, key=lambda r: r["cv_macro_f1_mean"], reverse=True)
    for r in sorted_results:
        print(
            f"{r['embedding_model']:<22} {r['classifier']:<10} "
            f"{r['cv_binary_f1_mean']:>7.4f} "
            f"{r['cv_macro_f1_mean']:>8.4f} "
            f"{r['cv_weighted_f1_mean']:>10.4f}"
        )

    best = sorted_results[0]
    print(f"\nBest: {best['embedding_model']} + {best['classifier']}")
    print(f"  macro F1:    {best['cv_macro_f1_mean']:.4f} ± {best['cv_macro_f1_std']:.4f}")
    print(f"  binary F1:   {best['cv_binary_f1_mean']:.4f} ± {best['cv_binary_f1_std']:.4f}")
    print(f"  weighted F1: {best['cv_weighted_f1_mean']:.4f} ± {best['cv_weighted_f1_std']:.4f}")
    rpt = best["oof_classification_report"]
    print("\nOOF classification report:")
    print(f"  {'':>6} {'precision':>10} {'recall':>8} {'f1':>8} {'support':>9}")
    for label in ["No", "Yes", "macro avg", "weighted avg"]:
        row = rpt[label]
        print(f"  {label:>12} {row['precision']:>10.2f} {row['recall']:>8.2f} {row['f1-score']:>8.2f} {row['support']:>9}")
    print(f"\nOOF confusion matrix:\n  {best['oof_confusion_matrix']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="results/qwen-embedding-8b-richard/manifest.json",
        help="Path to manifest.json listing parquet files.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save results JSON. Defaults to manifest's directory.",
    )
    args = parser.parse_args()

    manifest_path = pathlib.Path(args.manifest)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open() as f:
        manifest = json.load(f)

    print(f"Manifest: {manifest_path} ({len(manifest)} entries)")

    all_results: list[dict] = []

    for entry in manifest:
        model_name = entry["model_name"]
        parquet_path = entry["output_path"]
        print(f"\nLoading {parquet_path}")
        frame = load_frame(parquet_path)
        X, y = build_xy(frame)
        print(f"  shape={X.shape}, positive_ratio={y.mean():.3f}")

        for clf_name, clf in CLASSIFIERS.items():
            result = run_cv(X, y, clf, model_name, clf_name)
            all_results.append(result)

    print_summary(all_results)

    out_path = out_dir / "classifier_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
