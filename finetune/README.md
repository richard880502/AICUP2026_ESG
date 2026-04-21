
data
* pairwise_data (for cached_mnrl)
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

cached_mnrl
ls /mnt/share_data_79/embedd_ft_adapter/


### Visualization

```
python exp/exp1/compare_pair_similarity.py \
--pairs-path <training_data> \
--baseline-model <base_model> \
--finetune-model <finetuned_model>
```