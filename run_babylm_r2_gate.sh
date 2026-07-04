#!/usr/bin/env bash
# R2 -- gated attention (elementwise, use_attn_gate=True) on the winner d512/L24 + dropout 0.1.
# Adds ~n_embd^2/layer params (d512L24: 82.9M -> ~89.2M, +7.6% -- a capacity confound to note).
#   100M: +gate           (compare vs existing no-gate bl100m-d512L24-do0.1, seed 1337)
#   10M : no-gate + gate   (no 10M deep baseline exists yet -- run both)
# Winner arch RMS+SwiGLU 8/3+RoPE, dropout 0.1, sampler=shuffle, seed 1337 (default). head_dim=64.
#
# Usage: bash run_babylm_r2_gate.sh <maxit100> <warm100> <maxit10> <warm10> <batch> <gaccum>
#   e.g. bash run_babylm_r2_gate.sh 4740 100 466 40 32 16
PY=/media/volume/yupei-data/envs/nanogpt/bin/python
MI100="${1:?}"; W100="${2:?}"; MI10="${3:?}"; W10="${4:?}"; B="${5:-32}"; GA="${6:-16}"
DONE="results/r2gate.done"; rm -f "$DONE"

run () {  # name dataset maxit warm gate evalint
  local NAME="$1" DS="$2" MI="$3" WA="$4" GATE="$5" EV="$6"
  if grep -q "\"run_name\": \"${NAME}\"" results/experiments.jsonl 2>/dev/null; then
    echo "==== skip ${NAME} (already in experiments.jsonl) ===="; return; fi
  echo "================ ${NAME}  (gate=${GATE} dataset=${DS}) ================"
  "$PY" train.py config/train_babylm.py \
    --dataset="$DS" --max_iters="$MI" --lr_decay_iters="$MI" --warmup_iters="$WA" --eval_interval="$EV" \
    --use_rmsnorm=True --use_swiglu=True --use_rope=True \
    --sampler=shuffle --dropout=0.1 --use_attn_gate=$GATE \
    --n_embd=512 --n_layer=24 --n_head=8 \
    --batch_size="$B" --gradient_accumulation_steps="$GA" \
    --wandb_run_name="$NAME" --out_dir="out-babylm/${NAME}" \
    || echo "!!! FAILED: ${NAME}"
}

run bl100m-d512L24-do0.1-gate babylm_100m "$MI100" "$W100" True  500
run bl10m-d512L24-do0.1       babylm      "$MI10"  "$W10"  False 50
run bl10m-d512L24-do0.1-gate  babylm      "$MI10"  "$W10"  True  50
echo "==== R2 gate runs complete ===="; touch "$DONE"
