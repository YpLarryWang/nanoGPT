#!/usr/bin/env bash
# Single-GPU BabyLM 2026 AoA retraining with the required intermediate checkpoints.
# Usage: PY=/path/to/python CUDA_VISIBLE_DEVICES=0 bash run_babylm_aoa.sh {10m|100m}

set -euo pipefail

TRACK="${1:?usage: run_babylm_aoa.sh 10m-or-100m}"
PY="${PY:-python}"
B=32
GA=16

case "$TRACK" in
  10m)
    NAME=bl10m-d512L32-do0.1-gate-aoa19-b32ga16
    DATASET=babylm
    MAX_ITERS=466
    WARMUP_ITERS=40
    EVAL_INTERVAL=50
    SAVE_COUNT=19
    SAVE_ITERS='[5,9,14,19,23,28,33,37,42,47,93,140,186,233,280,326,373,419,466]'
    ;;
  100m)
    NAME=bl100m-d512L32-do0.1-gate-aoa28-b32ga16
    DATASET=babylm_100m
    MAX_ITERS=4740
    WARMUP_ITERS=100
    EVAL_INTERVAL=500
    SAVE_COUNT=28
    SAVE_ITERS='[5,9,14,19,24,28,33,38,43,47,95,142,190,237,284,332,379,427,474,948,1422,1896,2370,2844,3318,3792,4266,4740]'
    ;;
  *)
    echo "unknown track: $TRACK (expected 10m or 100m)" >&2
    exit 2
    ;;
esac

mkdir -p logs/aoa
echo "run=$NAME dataset=$DATASET batch=$B grad_accum=$GA eval_batch=32 seed=1337 save_count=$SAVE_COUNT"

exec "$PY" train.py config/train_babylm.py \
  --dataset="$DATASET" \
  --max_iters="$MAX_ITERS" --lr_decay_iters="$MAX_ITERS" \
  --warmup_iters="$WARMUP_ITERS" --eval_interval="$EVAL_INTERVAL" \
  --save_iters="$SAVE_ITERS" \
  --use_rmsnorm=True --use_swiglu=True --use_rope=True --use_attn_gate=True \
  --use_attn_res=False --use_muon=False \
  --sampler=shuffle --dropout=0.1 \
  --n_embd=512 --n_layer=32 --n_head=8 \
  --batch_size="$B" --gradient_accumulation_steps="$GA" \
  --eval_batch_size=32 \
  --seed=1337 --sampler_seed=1337 \
  --wandb_log=True --wandb_project=babylm \
  --wandb_run_name="$NAME" --out_dir="out-babylm/$NAME"
