import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


DEFAULT_PAIRS_PATH = Path("/tmp2/howard/side_project/ESG_AIDEA/finetune/contrastive_data/pairs.jsonl")
DEFAULT_BASELINE_MODEL = Path("/mnt/share_data_79/howard/models/qwen3-embedding-4b")
DEFAULT_FINETUNE_MODEL = Path("/tmp2/howard/side_project/ESG_AIDEA/finetune/models/triplet_lora_qwen/final_model")
DEFAULT_OUTPUT_DIR = Path("/tmp2/howard/side_project/ESG_AIDEA/finetune/exp/exp1/results/pair_similarity")
LABEL_NAME_MAP = {1: "should_close", 0: "should_far"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pair similarity between baseline and LoRA fine-tuned models.")
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--finetune-model", type=Path, default=DEFAULT_FINETUNE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use-cache", action="store_true")
    return parser.parse_args()


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        batch_size * torch.arange(batch_size, device=last_hidden_states.device) // batch_size,
        sequence_lengths,
    ]


def _triplets_to_pairs(rows: list[dict]) -> list[dict]:
    pair_rows = []
    for row in rows:
        pair_rows.append(
            {
                "text_a_id": row["anchor_id"],
                "text_b_id": row["positive_id"],
                "text_a": row["anchor"],
                "text_b": row["positive"],
                "label": 1,
            }
        )
        pair_rows.append(
            {
                "text_a_id": row["anchor_id"],
                "text_b_id": row["negative_id"],
                "text_a": row["anchor"],
                "text_b": row["negative"],
                "label": 0,
            }
        )
    return pair_rows


def load_pairs(pairs_path: Path) -> tuple[list[dict], pd.DataFrame]:
    raw_rows = []
    with pairs_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw_rows.append(json.loads(line))

    if not raw_rows:
        raise ValueError(f"Dataset is empty: {pairs_path}")

    first_row = raw_rows[0]
    if {"text_a_id", "text_b_id", "text_a", "text_b", "label"}.issubset(first_row):
        rows = raw_rows
    elif {"anchor_id", "positive_id", "negative_id", "anchor", "positive", "negative"}.issubset(first_row):
        rows = _triplets_to_pairs(raw_rows)
    else:
        raise ValueError(
            "Unsupported dataset format. Expected pair rows with "
            "`text_a/text_b/label` or triplet rows with `anchor/positive/negative`."
        )

    text_rows = []
    seen_ids = set()
    for row in rows:
        for text_id_key, text_key in (("text_a_id", "text_a"), ("text_b_id", "text_b")):
            text_id = row[text_id_key]
            if text_id in seen_ids:
                continue
            seen_ids.add(text_id)
            text_rows.append({"text_id": text_id, "text": row[text_key]})

    texts_df = pd.DataFrame(text_rows).sort_values("text_id").reset_index(drop=True)
    return rows, texts_df


def encode_texts(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        pooled = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        embeddings.append(pooled.cpu().float().numpy())
        print(f"  encoded {min(start + batch_size, len(texts))}/{len(texts)}")
    return np.vstack(embeddings)


def load_baseline_model(model_path: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


def load_finetune_model(base_model_path: Path, adapter_path: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    base_model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch_dtype)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.to(device)
    model.eval()
    return tokenizer, model


def build_similarity_df(
    pairs: list[dict],
    texts_df: pd.DataFrame,
    baseline_embeddings: np.ndarray,
    finetune_embeddings: np.ndarray,
) -> pd.DataFrame:
    baseline_map = dict(zip(texts_df["text_id"], baseline_embeddings))
    finetune_map = dict(zip(texts_df["text_id"], finetune_embeddings))

    records = []
    for row in pairs:
        text_a_id = row["text_a_id"]
        text_b_id = row["text_b_id"]
        baseline_similarity = float(np.dot(baseline_map[text_a_id], baseline_map[text_b_id]))
        finetune_similarity = float(np.dot(finetune_map[text_a_id], finetune_map[text_b_id]))
        records.append(
            {
                "text_a_id": text_a_id,
                "text_b_id": text_b_id,
                "label": row["label"],
                "label_name": LABEL_NAME_MAP[row["label"]],
                "text_a": row["text_a"],
                "text_b": row["text_b"],
                "baseline_similarity": baseline_similarity,
                "finetune_similarity": finetune_similarity,
                "delta_similarity": finetune_similarity - baseline_similarity,
            }
        )

    return pd.DataFrame(records)


def summarize(similarity_df: pd.DataFrame) -> dict:
    summary = {
        "overall": {
            "pair_count": int(len(similarity_df)),
            "baseline_mean_similarity": float(similarity_df["baseline_similarity"].mean()),
            "finetune_mean_similarity": float(similarity_df["finetune_similarity"].mean()),
            "mean_delta_similarity": float(similarity_df["delta_similarity"].mean()),
            "baseline_std_similarity": float(similarity_df["baseline_similarity"].std()),
            "finetune_std_similarity": float(similarity_df["finetune_similarity"].std()),
        },
        "by_label": {},
    }

    for label, group in similarity_df.groupby("label"):
        summary["by_label"][LABEL_NAME_MAP[label]] = {
            "pair_count": int(len(group)),
            "baseline_mean_similarity": float(group["baseline_similarity"].mean()),
            "finetune_mean_similarity": float(group["finetune_similarity"].mean()),
            "mean_delta_similarity": float(group["delta_similarity"].mean()),
            "baseline_std_similarity": float(group["baseline_similarity"].std()),
            "finetune_std_similarity": float(group["finetune_similarity"].std()),
        }

    return summary


def plot_distributions(similarity_df: pd.DataFrame, output_path: Path) -> None:
    bins = 50
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    should_close = similarity_df[similarity_df["label"] == 1]
    should_far = similarity_df[similarity_df["label"] == 0]

    axes[0].hist(should_close["baseline_similarity"], bins=bins, alpha=0.6, label="baseline", color="#1f77b4")
    axes[0].hist(should_close["finetune_similarity"], bins=bins, alpha=0.6, label="finetune", color="#d62728")
    axes[0].set_title('label="should_close"')
    axes[0].set_xlabel("Cosine similarity")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(should_far["baseline_similarity"], bins=bins, alpha=0.6, label="baseline", color="#1f77b4")
    axes[1].hist(should_far["finetune_similarity"], bins=bins, alpha=0.6, label="finetune", color="#d62728")
    axes[1].set_title('label="should_far"')
    axes[1].set_xlabel("Cosine similarity")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    boxplot_data = [
        similarity_df["baseline_similarity"].to_numpy(),
        similarity_df["finetune_similarity"].to_numpy(),
        should_close["finetune_similarity"].to_numpy(),
        should_far["finetune_similarity"].to_numpy(),
    ]
    axes[2].boxplot(
        boxplot_data,
        tick_labels=["baseline", "finetune", 'ft "should_close"', 'ft "should_far"'],
        showfliers=False,
    )
    axes[2].set_title("Similarity Spread")
    axes[2].set_ylabel("Cosine similarity")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def get_cache_paths(output_dir: Path) -> dict[str, Path]:
    cache_dir = output_dir / "cache"
    return {
        "cache_dir": cache_dir,
        "texts": cache_dir / "texts_with_ids.csv",
        "baseline": cache_dir / "baseline_embeddings.npy",
        "finetune": cache_dir / "finetune_embeddings.npy",
    }


def load_cached_embeddings(cache_paths: dict[str, Path]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray] | None:
    if not all(path.exists() for key, path in cache_paths.items() if key != "cache_dir"):
        return None
    texts_df = pd.read_csv(cache_paths["texts"], dtype={"text_id": str})
    baseline_embeddings = np.load(cache_paths["baseline"])
    finetune_embeddings = np.load(cache_paths["finetune"])
    return texts_df, baseline_embeddings, finetune_embeddings


def save_cached_embeddings(
    texts_df: pd.DataFrame,
    baseline_embeddings: np.ndarray,
    finetune_embeddings: np.ndarray,
    cache_paths: dict[str, Path],
) -> None:
    cache_paths["cache_dir"].mkdir(parents=True, exist_ok=True)
    texts_df.to_csv(cache_paths["texts"], index=False)
    np.save(cache_paths["baseline"], baseline_embeddings)
    np.save(cache_paths["finetune"], finetune_embeddings)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pairs from {args.pairs_path}")
    pairs, texts_df = load_pairs(args.pairs_path)
    print(f"Loaded {len(pairs)} pairs and {len(texts_df)} unique texts")

    cache_paths = get_cache_paths(args.output_dir)
    cached = load_cached_embeddings(cache_paths) if args.use_cache else None
    if cached is not None:
        print(f"Loading cached embeddings from {cache_paths['cache_dir']}")
        cached_texts_df, baseline_embeddings, finetune_embeddings = cached
        texts_df = cached_texts_df
    else:
        print(f"Using device: {args.device}")
        print(f"Loading baseline model from {args.baseline_model}")
        baseline_tokenizer, baseline_model = load_baseline_model(args.baseline_model, args.device)
        baseline_embeddings = encode_texts(
            baseline_model,
            baseline_tokenizer,
            texts_df["text"].tolist(),
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
        del baseline_model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        print(f"Loading finetune model from {args.finetune_model}")
        finetune_tokenizer, finetune_model = load_finetune_model(args.baseline_model, args.finetune_model, args.device)
        finetune_embeddings = encode_texts(
            finetune_model,
            finetune_tokenizer,
            texts_df["text"].tolist(),
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
        del finetune_model
        save_cached_embeddings(texts_df, baseline_embeddings, finetune_embeddings, cache_paths)

    similarity_df = build_similarity_df(pairs, texts_df, baseline_embeddings, finetune_embeddings)
    summary = summarize(similarity_df)

    similarity_csv_path = args.output_dir / "pair_similarity_scores.csv"
    summary_json_path = args.output_dir / "pair_similarity_summary.json"
    plot_path = args.output_dir / "pair_similarity_distribution.png"

    similarity_df.to_csv(similarity_csv_path, index=False)
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_distributions(similarity_df, plot_path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved scores to {similarity_csv_path}")
    print(f"Saved summary to {summary_json_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
