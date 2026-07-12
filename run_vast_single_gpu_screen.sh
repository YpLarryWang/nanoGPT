#!/usr/bin/env bash
# One single-GPU 10M screening run for the four-GPU Vast experiment matrix.
# Pin the GPU externally, e.g. CUDA_VISIBLE_DEVICES=2 bash run_vast_single_gpu_screen.sh kimi-attnres.

set -euo pipefail

VARIANT="${1:?usage: run_vast_single_gpu_screen.sh {kimi-base|kimi-attnres|muon-attnres}}"
PY="${PY:-/workspace/envs/nanogpt/bin/python}"
B="${B:-8}"
GA="${GA:-64}"
EXTRA=()

case "$VARIANT" in
  kimi-base)
    NAME=bl10m-d512L12-h4-do0.1-gate
    N_LAYER=12
    N_HEAD=4
    EXTRA+=(--use_attn_res=False --use_muon=False)
    ;;
  kimi-attnres)
    NAME=bl10m-d512L12-h4-do0.1-gate-attnres4
    N_LAYER=12
    N_HEAD=4
    EXTRA+=(--use_attn_res=True --attn_res_block_size=4 --use_muon=False)
    ;;
  muon-attnres)
    NAME=muon-bl10m-d512L32-do0.1-gate-attnres8
    N_LAYER=32
    N_HEAD=8
    EXTRA+=(--use_attn_res=True --attn_res_block_size=8 --use_muon=True)
    ;;
  *)
    echo "unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

mkdir -p logs/screens
echo "run=$NAME gpu=${CUDA_VISIBLE_DEVICES:-unset} batch=$B grad_accum=$GA seed=1337"

exec "$PY" train.py config/train_babylm.py \
  --dataset=babylm \
  --max_iters=466 --lr_decay_iters=466 --warmup_iters=40 --eval_interval=50 \
  --use_rmsnorm=True --use_swiglu=True --use_rope=True --use_attn_gate=True \
  "${EXTRA[@]}" \
  --sampler=shuffle --dropout=0.1 \
  --n_embd=512 --n_layer="$N_LAYER" --n_head="$N_HEAD" \
  --batch_size="$B" --gradient_accumulation_steps="$GA" \
  --seed=1337 --sampler_seed=1337 \
  --wandb_log=True --wandb_project=babylm \
  --wandb_run_name="$NAME" --out_dir="out-babylm/$NAME"
