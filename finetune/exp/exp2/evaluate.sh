#!/usr/bin/env bash
# Evaluate exp2 models: encode -> classify (inline, one model at a time)
# Run from finetune/ dir:
#   bash exp/exp2/evaluate.sh

set -e

CUDA_VISIBLE_DEVICES=1 python exp/exp2/inference.py
