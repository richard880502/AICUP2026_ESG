#!/usr/bin/env bash
# MNRL fine-tuning experiments
# Run from the finetune/ directory:
#   cd /tmp2/howard/side_project/ESG_AIDEA/finetune
#   bash exp/exp2/ft.sh

MODEL=/mnt/share_data_79/howard/models/qwen3-embedding-4b
TRAIN_PATH=contrastive_data/triplets_grouped.jsonl
BASE_CMD="python train_unsloth_triplet.py \
    --train-path $TRAIN_PATH \
    --model-name $MODEL \
    --loss-type mnrl \
    --trust-remote-code"

# ── exp2a: baseline ──────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 $BASE_CMD \
    --output-dir exp/exp2/exp2a \
    --lora-rank 16 --lora-alpha 32 \
    --per-device-batch-size 32 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-5 \
    --num-train-epochs 5

# ── exp2b: smaller batch (bs=16) ────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 $BASE_CMD \
    --output-dir exp/exp2/exp2b \
    --lora-rank 16 --lora-alpha 32 \
    --per-device-batch-size 16 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2e-5 \
    --num-train-epochs 5

# ── exp2c: bs=32 + larger LoRA rank ─────────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 $BASE_CMD \
    --output-dir exp/exp2/exp2c \
    --lora-rank 64 --lora-alpha 128 \
    --per-device-batch-size 32 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-5 \
    --num-train-epochs 5

# ── exp2d: bs=32 + higher LR + more epochs ──────────────────────────────────
CUDA_VISIBLE_DEVICES=1 $BASE_CMD \
    --output-dir exp/exp2/exp2d \
    --lora-rank 16 --lora-alpha 32 \
    --per-device-batch-size 32 \
    --gradient-accumulation-steps 4 \
    --learning-rate 5e-5 \
    --num-train-epochs 10

# ── exp2e: bs=32 + baseline LR + 50 epochs, save every 5 epochs (~285 steps) ─
CUDA_VISIBLE_DEVICES=1 $BASE_CMD \
    --output-dir exp/exp2/exp2e \
    --lora-rank 16 --lora-alpha 32 \
    --per-device-batch-size 32 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-5 \
    --num-train-epochs 50 \
    --save-steps 285 \
    --save-total-limit 11
