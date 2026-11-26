CUDA_VISIBLE_DEVICES=1 python train_unsloth_triplet.py \
    --train-path contrastive_data/triplets.jsonl \
    --output-dir models/triplet_lora_qwen \
    --model-name /mnt/share_data_79/howard/models/qwen3-embedding-4b/ \
    --trust-remote-code \
    --loss-type triplet \
    --lora-rank 16 --lora-alpha 32