"""
Inference: encode all records and save as parquet.
Usage:
  python inference.py --config configs/train/baseline.yml
  python inference.py --config configs/train/lora_exp1.yml
"""
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

SCRIPT_DIR = Path(__file__).parent
LORA_BASE_DIR = Path("/mnt/share_data_79/embedd_ft_adapter/")
DEFAULT_DATA_PATH = SCRIPT_DIR.parent.parent / "vpesg_4k_train_1000.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"


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


def encode_texts(texts: list[str], tokenizer, model, device: str, batch_size: int, max_length: int) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        embeddings = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        all_embeddings.append(embeddings.cpu().float().numpy())
        if (i // batch_size + 1) % 10 == 0:
            print(f"  encoded {min(i + batch_size, len(texts))}/{len(texts)}")
    return np.vstack(all_embeddings)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g. configs/train/baseline.yml)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    base_model_path = cfg["base_model_path"]
    lora_path_raw = cfg.get("lora_path")
    model_name = cfg["model_name"]
    batch_size = cfg.get("batch_size", 32)
    max_length = cfg.get("max_length", 512)

    data_path = Path(cfg["data_path"]) if cfg.get("data_path") else DEFAULT_DATA_PATH
    output_subdir = Path(cfg["output_dir"]) if cfg.get("output_dir") else DEFAULT_OUTPUT_DIR / model_name

    print(f"Loading data from {data_path}")
    with open(data_path) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading base model from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=dtype)

    if lora_path_raw:
        lora_path = Path(lora_path_raw)
        if not lora_path.is_absolute():
            lora_path = LORA_BASE_DIR / lora_path
        print(f"Loading LoRA adapter from {lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path))
        model = model.merge_and_unload()

    model.to(device)
    model.eval()

    texts = [r["data"] for r in records]
    print("Encoding texts...")
    embeddings = encode_texts(texts, tokenizer, model, device, batch_size, max_length)

    embedding_dim = embeddings.shape[1]
    print(f"Embedding dim: {embedding_dim}")

    rows = []
    for i, r in enumerate(records):
        text = r["data"]
        rows.append({
            "record_id": r["id"],
            "model_name": model_name,
            "text": text,
            "text_sha1": hashlib.sha1(text.encode()).hexdigest(),
            "embedding_dim": embedding_dim,
            "embedding": embeddings[i].tolist(),
            "promise_status": r.get("promise_status", ""),
            "promise_string": r.get("promise_string", ""),
            "verification_timeline": r.get("verification_timeline", ""),
            "evidence_status": r.get("evidence_status", ""),
            "evidence_quality": r.get("evidence_quality", ""),
            "esg_type": r.get("esg_type", ""),
            "company": r.get("company", ""),
            "ticker": r.get("ticker", ""),
            "page_number": r.get("page_number", ""),
        })

    df = pd.DataFrame(rows)

    output_subdir.mkdir(parents=True, exist_ok=True)
    output_path = output_subdir / "baseline.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Saved parquet to {output_path}")

    manifest_path = output_subdir / "manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    existing.append({
        "model_name": model_name,
        "rows": len(df),
        "embedding_dim": embedding_dim,
        "output_path": str(output_path.resolve()),
    })
    with open(manifest_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
