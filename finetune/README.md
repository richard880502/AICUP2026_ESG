
data
* pairwise_data (for cached_mnrl)
* explict negative (for mnrl)
* triplet_data (for triplet loss)

vanilla mode
```bash
CUDA_VISIBLE_DEVICES=1 python exp/exp1/inference.py
```

finetune
```
bash exp/exp1/ft.sh
CUDA_VISIBLE_DEVICES=1 python exp/exp1/inference.py --configs/eval/<adapter_config>.yml
```


評估
```bash
cd /tmp2/howard/side_project/ESG_AIDEA/finetune
CUDA_VISIBLE_DEVICES=1 .venv/bin/python inference.py --config configs/eval/cached_mnrl_lora_qwen_final_model.yml
CUDA_VISIBLE_DEVICES=1 .venv/bin/python classify.py --manifest results/exp2e/manifest.json 
```

### Visualization

```
python exp/exp1/compare_pair_similarity.py \
--pairs-path <training_data> \
--baseline-model <base_model> \
--finetune-model <finetuned_model>
```