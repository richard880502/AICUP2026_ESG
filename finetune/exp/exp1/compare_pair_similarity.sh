python exp/exp1/compare_pair_similarity.py \
 --pairs-path /mnt/share_data_79/contrastive_data_mixed/pairs.jsonl \
 --baseline-model /mnt/share_data_79/howard/models/qwen3-embedding-8b \
 --finetune-model /mnt/share_data_79/mixed_cached_mnrl/triplet_lora_qwen_mixed_cached_mnrl/final_model/ \
 --device cuda