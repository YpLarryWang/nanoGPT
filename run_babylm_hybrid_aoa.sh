#!/usr/bin/env bash
# One formal single-GPU Hybrid AoA run with exact word/BPE-token checkpoints.
#
# Usage:
#   PY=/path/to/python bash run_babylm_hybrid_aoa.sh TRACK SEED ATTNRES BATCH GRAD_ACCUM
#
# Examples:
#   bash run_babylm_hybrid_aoa.sh 10m 1337 0 32 16
#   bash run_babylm_hybrid_aoa.sh 10m 1338 0 32 16
#   bash run_babylm_hybrid_aoa.sh 10m 1337 1 32 16
#   bash run_babylm_hybrid_aoa.sh 100m 1337 0 32 16

set -euo pipefail

TRACK="${1:?TRACK must be 10m or 100m}"
SEED="${2:-1337}"
ATTNRES="${3:-0}"
B="${4:-32}"
GA="${5:-16}"
PY="${PY:-python}"

[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be an integer: $SEED" >&2; exit 2; }
[[ "$ATTNRES" = 0 || "$ATTNRES" = 1 ]] || { echo "ATTNRES must be 0 or 1" >&2; exit 2; }
(( B * GA == 512 )) || { echo "B*GA must equal 512, got $B*$GA" >&2; exit 2; }
(( GA % 16 == 0 )) || { echo "GA must be divisible by 16 to preserve the 15/16 Hybrid ratio" >&2; exit 2; }

CAUSAL=$((GA * 15 / 16))
MASKED=$((GA - CAUSAL))

case "$TRACK" in
  10m)
    DATASET=babylm
    MAX_ITERS=466
    WARMUP_ITERS=40
    EVAL_INTERVAL=50
    WORD_COUNT=18
    TOKEN_COUNT=20
    UNION_COUNT=36
    ;;
  100m)
    DATASET=babylm_100m
    MAX_ITERS=4740
    WARMUP_ITERS=100
    EVAL_INTERVAL=500
    WORD_COUNT=27
    TOKEN_COUNT=31
    UNION_COUNT=56
    ;;
  *)
    echo "unknown track: $TRACK (expected 10m or 100m)" >&2
    exit 2
    ;;
esac

SCHEDULE_STEM="bl${TRACK}-b${B}ga${GA}"
if [[ "$SEED" != 1337 ]]; then
  SCHEDULE_STEM+="-s${SEED}"
fi
CHECKPOINT_SCHEDULE="config/checkpoint_schedules/${SCHEDULE_STEM}-dual.json"
[[ -f "$CHECKPOINT_SCHEDULE" ]] || {
  echo "missing checkpoint schedule: $CHECKPOINT_SCHEDULE" >&2
  exit 2
}

NAME="bl${TRACK}-d512L32-do0.1-gate-hyb${CAUSAL}of${GA}"
ATTNRES_ARGS=(--use_attn_res=False)
if [[ "$ATTNRES" = 1 ]]; then
  NAME+="-attnres8"
  ATTNRES_ARGS=(--use_attn_res=True --attn_res_block_size=8)
fi
NAME+="-aoaw${WORD_COUNT}-aoat${TOKEN_COUNT}-u${UNION_COUNT}-b${B}ga${GA}"
if [[ "$SEED" != 1337 ]]; then
  NAME+="-s${SEED}"
fi

OUT="out-babylm/$NAME"
LOG="logs/aoa/$NAME.log"
DONE="results/$NAME.train.done"
FAILED="results/$NAME.train.failed"

[[ ! -e "$OUT" ]] || { echo "output already exists: $OUT" >&2; exit 1; }
if grep -q "\"run_name\": \"${NAME}\"" results/experiments.jsonl 2>/dev/null; then
  echo "run already exists in results/experiments.jsonl: $NAME" >&2
  exit 1
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  [[ -f "$HOME/.netrc" ]] && grep -q "machine api.wandb.ai" "$HOME/.netrc" || {
    echo "W&B authentication missing (neither WANDB_API_KEY nor api.wandb.ai in ~/.netrc)" >&2
    exit 1
  }
fi

mkdir -p logs/aoa results
rm -f "$DONE" "$FAILED"

echo "run=$NAME"
echo "dataset=$DATASET seed=$SEED sampler_seed=$SEED optimizer=adamw"
echo "hybrid=$CAUSAL causal/$MASKED masked batch=$B grad_accum=$GA tokens_per_update=$((B * GA * 512))"
echo "word_labels=$WORD_COUNT token_labels=$TOKEN_COUNT unique_ckpts=$UNION_COUNT schedule=$CHECKPOINT_SCHEDULE"
echo "log=$LOG"

exec "$PY" train.py config/train_babylm.py \
  --dataset="$DATASET" \
  --max_iters="$MAX_ITERS" --lr_decay_iters="$MAX_ITERS" \
  --warmup_iters="$WARMUP_ITERS" --eval_interval="$EVAL_INTERVAL" \
  --checkpoint_schedule="$CHECKPOINT_SCHEDULE" \
  --use_rmsnorm=True --use_swiglu=True --swiglu_mult=2.6666666666666665 \
  --use_rope=True --use_attn_gate=True \
  "${ATTNRES_ARGS[@]}" --use_muon=False \
  --use_hybrid=True --causal_microsteps="$CAUSAL" \
  --sampler=shuffle --sampler_seed="$SEED" --dropout=0.1 \
  --n_embd=512 --n_layer=32 --n_head=8 \
  --batch_size="$B" --gradient_accumulation_steps="$GA" \
  --eval_batch_size=32 \
  --seed="$SEED" \
  --wandb_log=True --wandb_project=babylm \
  --wandb_run_name="$NAME" --out_dir="$OUT"
