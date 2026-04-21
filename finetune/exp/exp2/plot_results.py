"""Plot classifier_results.json: F1 bar charts + OOF confusion matrices.

Usage (from finetune/ dir):
    python exp/exp2/plot_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

RESULTS_PATH = Path(__file__).parent / "results" / "classifier_results.json"
OUT_DIR = Path(__file__).parent / "results"

MODEL_ORDER = ["exp2-baseline", "exp2a", "exp2b", "exp2d"]
CLF_ORDER   = ["logreg", "xgboost"]
CLF_COLORS  = {"logreg": "#4C72B0", "xgboost": "#DD8452"}
METRICS     = ["cv_binary_f1_mean", "cv_macro_f1_mean", "cv_weighted_f1_mean"]
METRIC_LABELS = ["Binary F1", "Macro F1", "Weighted F1"]


def load(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    index: dict[tuple, dict] = {}
    for r in data:
        index[(r["embedding_model"], r["classifier"])] = r
    return index


# ── Figure 1: grouped bar chart ──────────────────────────────────────────────
def plot_f1_bars(index: dict, out_dir: Path) -> None:
    n_models  = len(MODEL_ORDER)
    n_metrics = len(METRICS)
    n_clf     = len(CLF_ORDER)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=False)
    fig.suptitle("CV F1 Scores by Model & Classifier", fontsize=14, fontweight="bold")

    x = np.arange(n_models)
    width = 0.35

    for ax, metric, label in zip(axes, METRICS, METRIC_LABELS):
        for i, clf in enumerate(CLF_ORDER):
            means = []
            stds  = []
            for model in MODEL_ORDER:
                r = index.get((model, clf), {})
                means.append(r.get(metric, 0))
                stds.append(r.get(metric.replace("mean", "std"), 0))
            offset = (i - (n_clf - 1) / 2) * width
            bars = ax.bar(x + offset, means, width, yerr=stds,
                          label=clf, color=CLF_COLORS[clf], alpha=0.85,
                          capsize=4, error_kw={"linewidth": 1.2})
            for bar, m in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{m:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_title(label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1 Score")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = out_dir / "f1_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ── Figure 2: confusion matrices ─────────────────────────────────────────────
def plot_confusion_matrices(index: dict, out_dir: Path) -> None:
    combos = [(m, c) for m in MODEL_ORDER for c in CLF_ORDER]
    n = len(combos)
    ncols = len(CLF_ORDER)
    nrows = len(MODEL_ORDER)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.2))
    fig.suptitle("OOF Confusion Matrices  (rows=Actual, cols=Predicted)",
                 fontsize=13, fontweight="bold")

    labels = ["No", "Yes"]

    for row_i, model in enumerate(MODEL_ORDER):
        for col_i, clf in enumerate(CLF_ORDER):
            ax = axes[row_i][col_i]
            r  = index.get((model, clf))
            if r is None:
                ax.axis("off")
                continue

            cm = np.array(r["oof_confusion_matrix"])
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

            for i in range(2):
                for j in range(2):
                    color = "white" if cm_norm[i, j] > 0.6 else "black"
                    ax.text(j, i,
                            f"{cm[i,j]}\n({cm_norm[i,j]:.1%})",
                            ha="center", va="center",
                            fontsize=11, color=color, fontweight="bold")

            ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
            ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("Actual", fontsize=9)

            macro_f1 = r["cv_macro_f1_mean"]
            ax.set_title(f"{model} | {clf}\nmacro F1={macro_f1:.3f}", fontsize=9)

    fig.tight_layout()
    out = out_dir / "confusion_matrices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


def main() -> None:
    index = load(RESULTS_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_f1_bars(index, OUT_DIR)
    plot_confusion_matrices(index, OUT_DIR)


if __name__ == "__main__":
    main()
