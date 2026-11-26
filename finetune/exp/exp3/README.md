# exp3 — Data Analysis

分析 `vpesg_4k_train_1000.json` 中各類句子在 **baseline** 與 **exp2d** 模型下的 embedding 相似度分布。

## 腳本說明

| 腳本 | 功能 |
|------|------|
| `promise_sim.py` | 計算每筆資料中 `promise_string` 與非 promise 部分的 cosine similarity |
| `plot_promise_sim.py` | 畫出上述 similarity 的分布圖（按 E/S/G 分類） |
| `promise_status_sim.py` | 計算 `promise_status=Yes` 與 `=No` 的句子之間的 pairwise similarity |

## 使用方式

所有腳本從 `finetune/` 目錄執行：

```bash
# 1. 計算 promise vs non-promise similarity（存 per-record sims）
CUDA_VISIBLE_DEVICES=1 python exp/exp3/promise_sim.py

# 2. 畫分布圖（需先執行步驟 1）
python exp/exp3/plot_promise_sim.py

# 3. 計算 promise_status Yes vs No pairwise similarity
CUDA_VISIBLE_DEVICES=1 python exp/exp3/promise_status_sim.py
```

## 輸出

```
results/
  promise_sim.json          # promise vs non-promise sim（含 per-record sims、E/S/G breakdown）
  promise_sim_dist.png      # 分布圖
  promise_status_sim.json   # yes_vs_yes / no_vs_no / yes_vs_no pairwise sim
```

## 模型路徑

- **baseline**: `qwen3-embedding-4b`（無 LoRA）
- **exp2d**: `exp/exp2/exp2d/final_model`
