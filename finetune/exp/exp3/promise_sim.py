"""Compute cosine similarity between promise_string and non-promise parts of data.

Usage (from finetune/ dir):
    CUDA_VISIBLE_DEVICES=1 python exp/exp3/promise_sim.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent
BASE_MODEL = "/mnt/share_data_79/howard/models/qwen3-embedding-4b"
DATA_PATH = Path(__file__).parent.parent.parent / "vpesg_4k_train_1000.json"
RESULTS_DIR = SCRIPT_DIR / "results"

MODELS: dict[str, Path | None] = {
    "baseline": None,
    "exp2d": SCRIPT_DIR.parent / "exp2" / "exp2d" / "final_model",
}


def last_token_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    left_pad = attn_mask[:, -1].sum() == attn_mask.shape[0]
    if left_pad:
        return last_hidden[:, -1]
    seq_lens = attn_mask.sum(dim=1) - 1
    bs = last_hidden.shape[0]
    return last_hidden[torch.arange(bs, device=last_hidden.device), seq_lens]


def encode(texts: list[str], tokenizer, model, device: str, batch_size: int = 32, max_length: int = 512) -> np.ndarray:
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


def get_non_promise(data: str, promise: str) -> str:
    return data.replace(promise, "").strip()


def main() -> None:
    with open(DATA_PATH) as f:
        records = json.load(f)

    # filter records that have a non-empty promise_string
    valid = [r for r in records if r.get("promise_string", "").strip()]
    print(f"Records with promise_string: {len(valid)} / {len(records)}")

    promise_texts = [r["promise_string"] for r in valid]
    non_promise_texts = [get_non_promise(r["data"], r["promise_string"]) for r in valid]
    # filter out cases where non-promise is empty
    pairs = [(p, n, r) for p, n, r in zip(promise_texts, non_promise_texts, valid) if n]
    promise_texts = [x[0] for x in pairs]
    non_promise_texts = [x[1] for x in pairs]
    valid = [x[2] for x in pairs]
    print(f"Valid pairs (non-empty non-promise): {len(valid)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    all_results = []

    for model_name, adapter_path in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        base = AutoModel.from_pretrained(BASE_MODEL, trust_remote_code=True, torch_dtype=dtype)

        if adapter_path is not None:
            print(f"  Loading LoRA from {adapter_path}")
            base = PeftModel.from_pretrained(base, str(adapter_path))
            base = base.merge_and_unload()

        base.to(device)
        base.eval()

        print(f"  Encoding promise_string ({len(promise_texts)})...")
        emb_promise = encode(promise_texts, tokenizer, base, device)
        print(f"  Encoding non_promise_string ({len(non_promise_texts)})...")
        emb_non_promise = encode(non_promise_texts, tokenizer, base, device)

        del base
        torch.cuda.empty_cache()

        sims = cosine_sim(emb_promise, emb_non_promise)

        esg_types = [r["esg_type"] for r in valid]
        result = {
            "model": model_name,
            "n": len(sims),
            "sim_mean": float(sims.mean()),
            "sim_std": float(sims.std()),
            "sim_min": float(sims.min()),
            "sim_max": float(sims.max()),
            "sims": sims.tolist(),
            "esg_types": esg_types,
        }
        for t in ["E", "S", "G"]:
            idx = [i for i, e in enumerate(esg_types) if e == t]
            if idx:
                s = sims[idx]
                result[f"{t}_mean"] = float(s.mean())
                result[f"{t}_std"] = float(s.std())
                result[f"{t}_n"] = len(idx)

        all_results.append(result)

        print(f"  sim mean={result['sim_mean']:.4f}  std={result['sim_std']:.4f}")
        for t in ["E", "S", "G"]:
            if f"{t}_mean" in result:
                print(f"    {t}: mean={result[f'{t}_mean']:.4f}  std={result[f'{t}_std']:.4f}  n={result[f'{t}_n']}")

    out_path = RESULTS_DIR / "promise_sim.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
