#!/usr/bin/env bash
# Formal single-GPU official-dev run with dual actual-word/BPE-token AoA checkpoints.
# Usage: PY=/path/to/python CUDA_VISIBLE_DEVICES=0 bash run_babylm_offdev_aoa.sh {10m|100m} [seed]

set -euo pipefail

TRACK="${1:?usage: run_babylm_offdev_aoa.sh 10m-or-100m [seed]}"
SEED="${2:-1337}"
PY="${PY:-python}"
B=32
GA=16

[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be an integer: $SEED" >&2; exit 2; }

case "$TRACK" in
  10m)
    DATASET=babylm_officialdev
    MAX_ITERS=471
    WARMUP_ITERS=40
    EVAL_INTERVAL=50
    WORD_COUNT=19
    TOKEN_COUNT=20
    UNION_COUNT=37
    ;;
  100m)
    DATASET=babylm_100m_officialdev
    MAX_ITERS=4797
    WARMUP_ITERS=100
    EVAL_INTERVAL=500
    WORD_COUNT=28
    TOKEN_COUNT=31
    UNION_COUNT=57
    ;;
  *)
    echo "unknown track: $TRACK (expected 10m or 100m)" >&2
    exit 2
    ;;
esac

SCHEDULE_STEM="bl${TRACK}-offdev-b${B}ga${GA}"
if [[ "$SEED" != 1337 ]]; then
  SCHEDULE_STEM+="-s${SEED}"
fi
CHECKPOINT_SCHEDULE="config/checkpoint_schedules/${SCHEDULE_STEM}-dual.json"
[[ -f "$CHECKPOINT_SCHEDULE" ]] || {
  echo "missing seed-specific checkpoint schedule: $CHECKPOINT_SCHEDULE" >&2
  exit 2
}

NAME="bl${TRACK}-d512L32-do0.1-gate-offdev-aoaw${WORD_COUNT}-aoat${TOKEN_COUNT}-u${UNION_COUNT}-b${B}ga${GA}"
if [[ "$SEED" != 1337 ]]; then
  NAME+="-s${SEED}"
fi

OUT="out-babylm/$NAME"
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
echo "run=$NAME"
echo "dataset=$DATASET seed=$SEED sampler_seed=$SEED optimizer=adamw"
echo "batch=$B grad_accum=$GA tokens_per_update=$((B * GA * 512))"
echo "word_labels=$WORD_COUNT token_labels=$TOKEN_COUNT unique_ckpts=$UNION_COUNT schedule=$CHECKPOINT_SCHEDULE"

exec "$PY" train.py config/train_babylm.py \
  --dataset="$DATASET" \
  --max_iters="$MAX_ITERS" --lr_decay_iters="$MAX_ITERS" \
  --warmup_iters="$WARMUP_ITERS" --eval_interval="$EVAL_INTERVAL" \
  --checkpoint_schedule="$CHECKPOINT_SCHEDULE" \
  --use_rmsnorm=True --use_swiglu=True --swiglu_mult=2.6666666666666665 \
  --use_rope=True --use_attn_gate=True \
  --use_attn_res=False --use_muon=False --use_hybrid=False \
  --sampler=shuffle --sampler_seed="$SEED" --dropout=0.1 \
  --n_embd=512 --n_layer=32 --n_head=8 \
  --batch_size="$B" --gradient_accumulation_steps="$GA" \
  --eval_batch_size=32 \
  --seed="$SEED" \
  --wandb_log=True --wandb_project=babylm \
  --wandb_run_name="$NAME" --out_dir="$OUT"
