#!/usr/bin/env bash
# Serial AttnRes verdict run: 10M smoke first, then the 100M headline run.
# Jetstream is a 20GB MIG slice, so B=8 / GA=64 preserves 262,144 tokens/iter
# while leaving headroom for the depth-ledger activations.

set -uo pipefail

PY=/media/volume/yupei-data/envs/nanogpt/bin/python
B="${1:-8}"
GA="${2:-64}"
LOG_DIR=logs/attnres
mkdir -p "$LOG_DIR"

run_variant() { # name dataset max_iters warmup eval_interval
  local NAME="$1" DS="$2" MAXIT="$3" WARM="$4" EVALINT="$5"
  local GPU_LOG="$LOG_DIR/${NAME}.gpu.csv"

  if grep -q "\"run_name\": \"${NAME}\"" results/experiments.jsonl 2>/dev/null; then
    echo "==== skip ${NAME} (already in experiments.jsonl) ===="
    return 0
  fi

  echo "================ ${NAME} ================"
  echo "dataset=${DS} max_iters=${MAXIT} batch=${B} grad_accum=${GA} seed=1337"
  echo "timestamp,memory_used_MiB,gpu_util_percent" > "$GPU_LOG"

  "$PY" train.py config/train_babylm.py \
    --dataset="$DS" \
    --max_iters="$MAXIT" --lr_decay_iters="$MAXIT" --warmup_iters="$WARM" --eval_interval="$EVALINT" \
    --use_rmsnorm=True --use_swiglu=True --use_rope=True --use_attn_gate=True \
    --use_attn_res=True --attn_res_block_size=8 \
    --sampler=shuffle --dropout=0.1 \
    --n_embd=512 --n_layer=32 --n_head=8 \
    --batch_size="$B" --gradient_accumulation_steps="$GA" \
    --seed=1337 --sampler_seed=1337 \
    --wandb_log=True --wandb_project=babylm \
    --wandb_run_name="$NAME" --out_dir="out-babylm/${NAME}" &
  local TRAIN_PID=$!

  (
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
      printf '%s,' "$(date --iso-8601=seconds)" >> "$GPU_LOG"
      nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits >> "$GPU_LOG"
      sleep 5
    done
  ) &
  local MONITOR_PID=$!

  wait "$TRAIN_PID"
  local STATUS=$?
  wait "$MONITOR_PID" 2>/dev/null || true
  if [[ "$STATUS" -ne 0 ]]; then
    echo "!!!! ${NAME} failed with status ${STATUS}; stopping serial queue !!!!"
    return "$STATUS"
  fi
  echo "==== completed ${NAME} ===="
}

run_variant bl10m-d512L32-do0.1-gate-attnres8 babylm      466  40  50  || exit $?
run_variant bl100m-d512L32-do0.1-gate-attnres8 babylm_100m 4740 100 500 || exit $?

echo "==== AttnRes 10M -> 100M serial queue complete ===="
