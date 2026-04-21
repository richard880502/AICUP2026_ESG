"""Encode classification data with each exp2 model, then immediately classify.

Encode one model → run CV classifier → free GPU → next model.

Usage (from finetune/ dir):
    CUDA_VISIBLE_DEVICES=1 python exp/exp2/inference.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from classify import CLASSIFIERS, build_xy, load_frame, print_summary, run_cv

SCRIPT_DIR = Path(__file__).parent
BASE_MODEL = "/mnt/share_data_79/howard/models/qwen3-embedding-4b"
DEFAULT_DATA = Path(__file__).parent.parent.parent / "vpesg_4k_train_1000.json"
RESULTS_DIR = SCRIPT_DIR / "results"

MODELS: dict[str, Path | None] = {
    "exp2-baseline": None,
    "exp2a": SCRIPT_DIR / "exp2a" / "final_model",
    "exp2b": SCRIPT_DIR / "exp2b" / "final_model",
    "exp2d": SCRIPT_DIR / "exp2d" / "final_model",
}


def last_token_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    left_pad = attn_mask[:, -1].sum() == attn_mask.shape[0]
    if left_pad:
        return last_hidden[:, -1]
    seq_lens = attn_mask.sum(dim=1) - 1
    bs = last_hidden.shape[0]
    return last_hidden[torch.arange(bs, device=last_hidden.device), seq_lens]


def encode(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
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
        done = min(i + batch_size, len(texts))
        if (i // batch_size + 1) % 10 == 0:
            print(f"    encoded {done}/{len(texts)}")
    return np.vstack(all_embs)


def encode_and_save(
    model_name: str,
    adapter_path: Path | None,
    records: list[dict],
    device: str,
    dtype: torch.dtype,
    batch_size: int,
    max_length: int,
    out_dir: Path,
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

    texts = [r["data"] for r in records]
    t0 = time.time()
    print(f"  Encoding {len(texts)} texts...")
    embeddings = encode(texts, tokenizer, base, device, batch_size, max_length)
    elapsed = round(time.time() - t0, 1)
    print(f"  Done in {elapsed}s, dim={embeddings.shape[1]}")

    del base
    torch.cuda.empty_cache()

    rows = []
    for i, r in enumerate(records):
        text = r["data"]
        rows.append({
            "record_id": r["id"],
            "model_name": model_name,
            "text": text,
            "text_sha1": hashlib.sha1(text.encode()).hexdigest(),
            "embedding_dim": embeddings.shape[1],
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
    parquet_path = out_dir / f"{model_name}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"  Saved parquet → {parquet_path}")

    return {
        "model_name": model_name,
        "rows": len(df),
        "embedding_dim": int(embeddings.shape[1]),
        "output_path": str(parquet_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(DEFAULT_DATA))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    print(f"Loading data from {data_path}")
    with open(data_path) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    all_clf_results: list[dict] = []

    for model_name, adapter_path in MODELS.items():
        if adapter_path is not None and not adapter_path.exists():
            print(f"Skipping {model_name}: adapter not found at {adapter_path}")
            continue

        # --- encode ---
        entry = encode_and_save(
            model_name, adapter_path, records,
            device, dtype, args.batch_size, args.max_length,
            RESULTS_DIR,
        )
        manifest.append(entry)

        # --- classify immediately after GPU is freed ---
        print(f"\n  Running classifiers on {model_name}...")
        frame = load_frame(entry["output_path"])
        X, y = build_xy(frame)
        print(f"  shape={X.shape}, positive_ratio={y.mean():.3f}")
        for clf_name, clf in CLASSIFIERS.items():
            result = run_cv(X, y, clf, model_name, clf_name)
            all_clf_results.append(result)

    # save manifest
    manifest_path = RESULTS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest → {manifest_path}")

    # print + save classifier results
    print_summary(all_clf_results)
    clf_out = RESULTS_DIR / "classifier_results.json"
    with open(clf_out, "w", encoding="utf-8") as f:
        json.dump(all_clf_results, f, indent=2, ensure_ascii=False)
    print(f"Saved classifier results → {clf_out}")


if __name__ == "__main__":
    main()
