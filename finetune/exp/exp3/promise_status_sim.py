"""Compute similarity between promise_status=Yes and promise_status=No sentences.

Encodes all `data` texts, then computes:
  - yes_vs_yes: mean pairwise cosine sim within Yes group (all)
  - no_vs_no:   mean pairwise cosine sim within No group (all)
  - yes_vs_no:  mean pairwise cosine sim between Yes and No groups (all)

Usage (from finetune/ dir):
    CUDA_VISIBLE_DEVICES=1 python exp/exp3/promise_status_sim.py
"""
from __future__ import annotations

import json
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


def pairwise_mean_sim(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean and std of all pairwise cosine sims between rows of a and b."""
    sims = a @ b.T  # (n, m)
    vals = sims.flatten()
    return float(vals.mean()), float(vals.std())


def self_pairwise_mean_sim(a: np.ndarray) -> tuple[float, float]:
    """Mean and std of upper-triangle pairwise cosine sims within a."""
    sims = a @ a.T
    n = sims.shape[0]
    idx = np.triu_indices(n, k=1)
    vals = sims[idx]
    return float(vals.mean()), float(vals.std())


def main() -> None:
    with open(DATA_PATH) as f:
        records = json.load(f)

    yes_records = [r for r in records if r.get("promise_status") == "Yes"]
    no_records  = [r for r in records if r.get("promise_status") == "No"]
    print(f"Yes: {len(yes_records)}, No: {len(no_records)}, Total: {len(records)}")

    yes_texts = [r["data"] for r in yes_records]
    no_texts  = [r["data"] for r in no_records]
    all_texts = [r["data"] for r in records]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    all_results = []

    for model_name, adapter_path in MODELS.items():
        print(f"\n{'='*50}\nModel: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        base = AutoModel.from_pretrained(BASE_MODEL, trust_remote_code=True, torch_dtype=dtype)

        if adapter_path is not None:
            print(f"  Loading LoRA from {adapter_path}")
            base = PeftModel.from_pretrained(base, str(adapter_path))
            base = base.merge_and_unload()

        base.to(device)
        base.eval()

        print(f"  Encoding Yes ({len(yes_texts)})...")
        emb_yes = encode(yes_texts, tokenizer, base, device)
        print(f"  Encoding No ({len(no_texts)})...")
        emb_no  = encode(no_texts,  tokenizer, base, device)
        print(f"  Encoding All ({len(all_texts)})...")
        emb_all = encode(all_texts, tokenizer, base, device)

        del base
        torch.cuda.empty_cache()

        yy_sims  = (emb_yes @ emb_yes.T)[np.triu_indices(len(emb_yes), k=1)].tolist()
        nn_sims  = (emb_no  @ emb_no.T )[np.triu_indices(len(emb_no),  k=1)].tolist()
        yn_sims  = (emb_yes @ emb_no.T ).flatten().tolist()
        all_sims = (emb_all @ emb_all.T)[np.triu_indices(len(emb_all), k=1)].tolist()

        def stats(s): return {"mean": float(np.mean(s)), "std": float(np.std(s)), "sims": s}

        result = {
            "model": model_name,
            "yes_n": len(yes_texts),
            "no_n":  len(no_texts),
            "all_n": len(all_texts),
            "yes_vs_yes": stats(yy_sims),
            "no_vs_no":   stats(nn_sims),
            "yes_vs_no":  stats(yn_sims),
            "any_vs_any": stats(all_sims),
        }
        all_results.append(result)

        for k in ["yes_vs_yes", "no_vs_no", "yes_vs_no", "any_vs_any"]:
            print(f"  {k}: {result[k]['mean']:.4f} ± {result[k]['std']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "promise_status_sim.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
