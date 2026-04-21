1. 用 mnrl loss 訓練
2. 超參數調整 (epoch、batch size、)
3. 評估 (embedding + classifier)
4. 計算距離

```bash
bash ft.sh
bash evaluate.sh

CUDA_VISIBLE_DEVICES=1 python exp/exp2/triplet_distance.py
Loading triplets from /tmp2/howard/side_project/ESG_AIDEA/finetune/contrastive_data/triplets_grouped.jsonl
```


## 分析
看證據與非證據的距離
ESG 彼此的距離