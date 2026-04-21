"""Plot triplet distance results.

If triplet_distance_samples.parquet exists (produced by triplet_distance.py),
uses real per-sample distributions. Otherwise falls back to summary JSON only
and approximates with normal distributions.

Usage (from finetune/ dir):
    python exp/exp2/plot_triplet.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR  = Path(__file__).parent / "results"
JSON_PATH    = RESULTS_DIR / "triplet_distance.json"
PARQUET_PATH = RESULTS_DIR / "triplet_distance_samples.parquet"

MODEL_ORDER  = ["baseline", "exp2a", "exp2b", "exp2d"]
MODEL_COLORS = {"baseline": "#888888", "exp2a": "#4C72B0", "exp2b": "#55A868", "exp2d": "#DD8452"}


def load_data():
    with open(JSON_PATH) as f:
        summary = json.load(f)
    summary_index = {r["model"]: r for r in summary}

    samples = None
    if PARQUET_PATH.exists():
        import pandas as pd
        samples = pd.read_parquet(PARQUET_PATH)
        print(f"Loaded per-sample parquet: {len(samples)} rows")
    else:
        print("No parquet found — using normal approximation from summary stats")

    return summary_index, samples


def get_arrays(model: str, summary_index: dict, samples) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if samples is not None:
        sub = samples[samples["model"] == model]
        return sub["pos_sim"].values, sub["neg_sim"].values, sub["gap"].values
    # approximate with normal
    r    = summary_index[model]
    n    = r["n"]
    rng  = np.random.default_rng(42)
    pos  = rng.normal(r["pos_sim_mean"], r["pos_sim_std"], n).clip(-1, 1)
    neg  = rng.normal(r["neg_sim_mean"], r["neg_sim_std"], n).clip(-1, 1)
    gap  = pos - neg
    return pos.astype(np.float32), neg.astype(np.float32), gap.astype(np.float32)


def plot_violin(summary_index: dict, samples, out_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in summary_index]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Anchor↔Positive vs Anchor↔Negative Cosine Similarity\n(violin = distribution, ◆ = mean ± std)",
                 fontsize=12, fontweight="bold")

    configs = [
        ("pos_sim", "pos_sim_mean", "pos_sim_std", "Positive Similarity"),
        ("neg_sim", "neg_sim_mean", "neg_sim_std", "Negative Similarity"),
        ("gap",     "gap_mean",     "gap_std",      "Gap  (pos − neg)"),
    ]

    for ax, (arr_key, mean_key, std_key, title) in zip(axes, configs):
        data = []
        for m in models:
            pos, neg, gap = get_arrays(m, summary_index, samples)
            data.append({"pos_sim": pos, "neg_sim": neg, "gap": gap}[arr_key])

        parts = ax.violinplot(data, positions=range(len(models)),
                              showmedians=True, showextrema=False)
        for pc, model in zip(parts["bodies"], models):
            pc.set_facecolor(MODEL_COLORS.get(model, "#999"))
            pc.set_alpha(0.72)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

        for i, model in enumerate(models):
            r = summary_index[model]
            ax.errorbar(i, r[mean_key], yerr=r[std_key],
                        fmt="D", color="crimson", markersize=5, linewidth=1.5, zorder=5)

        if arr_key == "gap":
            ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.6)

        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel("Cosine Similarity")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    p = out_dir / "triplet_violin.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved → {p}")
    plt.close(fig)


def plot_gap_kde(summary_index: dict, samples, out_dir: Path) -> None:
    try:
        from scipy.stats import gaussian_kde
        has_scipy = True
    except ImportError:
        has_scipy = False

    models = [m for m in MODEL_ORDER if m in summary_index]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Gap Distribution  (pos_sim − neg_sim)  per Model", fontsize=13, fontweight="bold")

    for model in models:
        _, _, gap = get_arrays(model, summary_index, samples)
        color = MODEL_COLORS.get(model, "#999")
        acc   = summary_index[model]["accuracy"]
        label = f"{model}  (acc={acc:.3f})"
        if has_scipy:
            kde = gaussian_kde(gap, bw_method=0.25)
            xs  = np.linspace(gap.min() - 0.05, gap.max() + 0.05, 400)
            ax.plot(xs, kde(xs), label=label, color=color, linewidth=2)
            ax.fill_between(xs, kde(xs), alpha=0.15, color=color)
        else:
            ax.hist(gap, bins=60, density=True, alpha=0.4, color=color, label=label)

    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="gap = 0")
    ax.set_xlabel("Gap  (pos_sim − neg_sim)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    p = out_dir / "triplet_gap_kde.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved → {p}")
    plt.close(fig)


def plot_scatter(summary_index: dict, samples, out_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in summary_index]
    n      = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.8))
    if n == 1:
        axes = [axes]
    fig.suptitle("Positive vs Negative Similarity  (each dot = one triplet)",
                 fontsize=12, fontweight="bold")

    for ax, model in zip(axes, models):
        pos, neg, gap = get_arrays(model, summary_index, samples)
        color = MODEL_COLORS.get(model, "#999")
        above = pos > neg
        ax.scatter(neg[above],  pos[above],  c=color,    alpha=0.22, s=5,
                   label=f"pos wins ({above.mean():.1%})")
        ax.scatter(neg[~above], pos[~above], c="#e74c3c", alpha=0.35, s=5,
                   label=f"neg wins ({(~above).mean():.1%})")
        lo = min(pos.min(), neg.min()) - 0.03
        hi = max(pos.max(), neg.max()) + 0.03
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("neg_sim"); ax.set_ylabel("pos_sim")
        ax.set_title(f"{model}\nacc={summary_index[model]['accuracy']:.3f}", fontsize=10)
        ax.legend(fontsize=7.5, markerscale=3)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    p = out_dir / "triplet_scatter.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved → {p}")
    plt.close(fig)


def plot_summary_bar(summary_index: dict, out_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in summary_index]
    metrics = [
        ("gap_mean",  "gap_std",  "Mean Gap  (pos − neg)"),
        ("accuracy",  None,       "Accuracy  (pos closer than neg)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Triplet Retrieval Quality per Model", fontsize=13, fontweight="bold")

    x = np.arange(len(models))
    for ax, (mkey, skey, title) in zip(axes, metrics):
        vals  = [summary_index[m][mkey] for m in models]
        errs  = [summary_index[m][skey] for m in models] if skey else None
        colors = [MODEL_COLORS.get(m, "#999") for m in models]
        bars = ax.bar(x, vals, yerr=errs, color=colors, alpha=0.85,
                      capsize=5, error_kw={"linewidth": 1.5})
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.003 if errs is None else max(errs) * 0.1),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(mkey.replace("_", " "))
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    p = out_dir / "triplet_summary_bar.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved → {p}")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_index, samples = load_data()

    plot_summary_bar(summary_index, RESULTS_DIR)
    plot_violin(summary_index, samples, RESULTS_DIR)
    plot_gap_kde(summary_index, samples, RESULTS_DIR)
    plot_scatter(summary_index, samples, RESULTS_DIR)


if __name__ == "__main__":
    main()
