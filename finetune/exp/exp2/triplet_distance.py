"""Compare anchor↔positive vs anchor↔negative cosine similarity before/after fine-tuning.

Usage (from finetune/ dir):
    CUDA_VISIBLE_DEVICES=1 python exp/exp2/triplet_distance.py
    CUDA_VISIBLE_DEVICES=1 python exp/exp2/triplet_distance.py --n 200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent
BASE_MODEL = "/mnt/share_data_79/howard/models/qwen3-embedding-4b"
TRIPLETS_PATH = Path(__file__).parent.parent.parent / "contrastive_data" / "triplets_grouped.jsonl"

MODELS: dict[str, Path | None] = {
    "baseline": None,
    "exp2a":    SCRIPT_DIR / "exp2a" / "final_model",
    "exp2b":    SCRIPT_DIR / "exp2b" / "final_model",
    "exp2d":    SCRIPT_DIR / "exp2d" / "final_model",
    "exp2e":    SCRIPT_DIR / "exp2e" / "final_model",
}


def last_token_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    left_pad = attn_mask[:, -1].sum() == attn_mask.shape[0]
    if left_pad:
        return last_hidden[:, -1]
    seq_lens = attn_mask.sum(dim=1) - 1
    bs = last_hidden.shape[0]
    return last_hidden[torch.arange(bs, device=last_hidden.device), seq_lens]


def encode(texts: list[str], tokenizer, model, device: str, batch_size: int, max_length: int) -> np.ndarray:
    all_embs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        emb = last_token_pool(out.last_hidden_state, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        all_embs.append(emb.cpu().float().numpy())
    return np.vstack(all_embs)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # both already L2-normalized
    return (a * b).sum(axis=1)


def evaluate_model(
    model_name: str,
    adapter_path: Path | None,
    anchors: list[str],
    positives: list[str],
    negatives: list[str],
    device: str,
    dtype: torch.dtype,
    batch_size: int,
    max_length: int,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModel.from_pretrained(BASE_MODEL, trust_remote_code=True, torch_dtype=dtype)

    if adapter_path is not None:
        print(f"  Loading LoRA from {adapter_path}")
        base = PeftModel.from_pretrained(base, str(adapter_path))
        base = base.merge_and_unload()

    base.to(device)
    base.eval()

    t0 = time.time()
    print(f"  Encoding {len(anchors)} anchors...")
    anc_emb = encode(anchors,   tokenizer, base, device, batch_size, max_length)
    pos_emb = encode(positives, tokenizer, base, device, batch_size, max_length)
    neg_emb = encode(negatives, tokenizer, base, device, batch_size, max_length)
    print(f"  Done in {round(time.time()-t0,1)}s")

    del base
    torch.cuda.empty_cache()

    pos_sim = cosine_sim(anc_emb, pos_emb)
    neg_sim = cosine_sim(anc_emb, neg_emb)
    gap     = pos_sim - neg_sim
    acc     = (gap > 0).mean()

    result = {
        "model": model_name,
        "n": len(anchors),
        "pos_sim_mean": float(pos_sim.mean()),
        "pos_sim_std":  float(pos_sim.std()),
        "neg_sim_mean": float(neg_sim.mean()),
        "neg_sim_std":  float(neg_sim.std()),
        "gap_mean":     float(gap.mean()),
        "gap_std":      float(gap.std()),
        "accuracy":     float(acc),
        "_pos_sim": pos_sim,
        "_neg_sim": neg_sim,
        "_gap":     gap,
    }

    print(f"  pos_sim : {result['pos_sim_mean']:.4f} ± {result['pos_sim_std']:.4f}")
    print(f"  neg_sim : {result['neg_sim_mean']:.4f} ± {result['neg_sim_std']:.4f}")
    print(f"  gap     : {result['gap_mean']:.4f} ± {result['gap_std']:.4f}")
    print(f"  accuracy: {result['accuracy']:.4f}  (anchor closer to pos than neg)")

    return result


MODEL_ORDER  = ["baseline", "exp2a", "exp2b", "exp2d", "exp2e"]
MODEL_COLORS = {"baseline": "#888888", "exp2a": "#4C72B0", "exp2b": "#55A868", "exp2d": "#DD8452", "exp2e": "#C44E52"}


def plot_distributions(all_results: list[dict], out_dir: Path) -> None:
    models = [r["model"] for r in all_results]

    # ── Fig 1: pos/neg sim distributions (violin) ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Anchor↔Positive vs Anchor↔Negative Cosine Similarity", fontsize=13, fontweight="bold")

    titles = ["Positive Similarity", "Negative Similarity", "Gap (pos − neg)"]
    keys   = ["_pos_sim", "_neg_sim", "_gap"]

    for ax, key, title in zip(axes, keys, titles):
        data = [r[key] for r in all_results]
        parts = ax.violinplot(data, positions=range(len(models)),
                              showmedians=True, showextrema=False)
        for i, (pc, model) in enumerate(zip(parts["bodies"], models)):
            pc.set_facecolor(MODEL_COLORS.get(model, "#999"))
            pc.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

        # overlay mean ± std
        for i, r in enumerate(all_results):
            mean_key = key.lstrip("_") + "_mean"
            std_key  = key.lstrip("_") + "_std"
            if mean_key in r:
                ax.errorbar(i, r[mean_key], yerr=r[std_key],
                            fmt="D", color="crimson", markersize=5, linewidth=1.5, zorder=5)

        if key == "_gap":
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

    # ── Fig 2: gap KDE per model on one axis ─────────────────────────────────
    try:
        from scipy.stats import gaussian_kde
        has_scipy = True
    except ImportError:
        has_scipy = False

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Gap Distribution (pos_sim − neg_sim) per Model", fontsize=13, fontweight="bold")

    for r in all_results:
        model = r["model"]
        gap   = r["_gap"]
        color = MODEL_COLORS.get(model, "#999")
        if has_scipy and len(gap) > 1:
            kde = gaussian_kde(gap, bw_method=0.3)
            xs  = np.linspace(gap.min() - 0.05, gap.max() + 0.05, 300)
            ax.plot(xs, kde(xs), label=f"{model} (acc={r['accuracy']:.3f})", color=color, linewidth=2)
            ax.fill_between(xs, kde(xs), alpha=0.15, color=color)
        else:
            bins = min(60, max(1, len(gap)))
            ax.hist(gap, bins=bins, density=True, alpha=0.4, color=color,
                    label=f"{model} (acc={r['accuracy']:.3f})")

    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="gap=0")
    ax.set_xlabel("Gap (pos_sim − neg_sim)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    p = out_dir / "triplet_gap_kde.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved → {p}")
    plt.close(fig)

    # ── Fig 3: scatter pos_sim vs neg_sim per model ───────────────────────────
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]
    fig.suptitle("Positive vs Negative Similarity (per triplet)", fontsize=13, fontweight="bold")

    for ax, r in zip(axes, all_results):
        model = r["model"]
        pos   = r["_pos_sim"]
        neg   = r["_neg_sim"]
        color = MODEL_COLORS.get(model, "#999")
        above = pos > neg
        ax.scatter(neg[above],  pos[above],  c=color,    alpha=0.25, s=6, label=f"pos wins ({above.mean():.1%})")
        ax.scatter(neg[~above], pos[~above], c="#e74c3c", alpha=0.35, s=6, label=f"neg wins ({(~above).mean():.1%})")
        lim_min = min(pos.min(), neg.min()) - 0.02
        lim_max = max(pos.max(), neg.max()) + 0.02
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1, alpha=0.5)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel("neg_sim")
        ax.set_ylabel("pos_sim")
        ax.set_title(f"{model}\nacc={r['accuracy']:.3f}", fontsize=10)
        ax.legend(fontsize=7.5, markerscale=3)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    p = out_dir / "triplet_scatter.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved → {p}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(TRIPLETS_PATH))
    parser.add_argument("--n",         type=int, default=None, help="Use first N triplets (None = all)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    print(f"Loading triplets from {args.data_path}")
    with open(args.data_path) as f:
        rows = [json.loads(l) for l in f]
    if args.n:
        rows = rows[: args.n]
    print(f"Using {len(rows)} triplets")

    anchors   = [r["anchor"]   for r in rows]
    positives = [r["positive"] for r in rows]
    # each row may have multiple negatives — use the first one
    negatives = [r["negatives"][0] for r in rows]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    all_results: list[dict] = []

    for model_name, adapter_path in MODELS.items():
        if adapter_path is not None and not adapter_path.exists():
            print(f"Skipping {model_name}: adapter not found at {adapter_path}")
            continue
        result = evaluate_model(
            model_name, adapter_path,
            anchors, positives, negatives,
            device, dtype, args.batch_size, args.max_length,
        )
        all_results.append(result)

    # --- summary table ---
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    hdr = f"{'model':<14} {'pos_sim':>8} {'neg_sim':>8} {'gap':>8} {'accuracy':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in all_results:
        print(
            f"{r['model']:<14} "
            f"{r['pos_sim_mean']:>8.4f} "
            f"{r['neg_sim_mean']:>8.4f} "
            f"{r['gap_mean']:>8.4f} "
            f"{r['accuracy']:>10.4f}"
        )

    out_path = SCRIPT_DIR / "results" / "triplet_distance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_results]
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")

    # --- per-sample parquet ---
    dfs = []
    for r in all_results:
        n = len(r["_gap"])
        dfs.append(pd.DataFrame({
            "model":   r["model"],
            "pos_sim": r["_pos_sim"],
            "neg_sim": r["_neg_sim"],
            "gap":     r["_gap"],
        }))
    df_all = pd.concat(dfs, ignore_index=True)
    pq_path = SCRIPT_DIR / "results" / "triplet_distance_samples.parquet"
    df_all.to_parquet(pq_path, index=False)
    print(f"Saved → {pq_path}")

    plot_distributions(all_results, SCRIPT_DIR / "results")


if __name__ == "__main__":
    main()
