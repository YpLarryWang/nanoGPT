#!/usr/bin/env bash
# Phase C -- multi-seed the 83M WINNER (d512/L24 + dropout 0.1 = "deep+dropout"), which beat the 98M
# GPT-2 baseline on avg5 (56.04 vs 54.71). Goal: error bars on that headline + on the noisy
# entity_tracking task. Existing Phase-B run is seed 1337; this adds 1338, 1339 -> n=3.
# Each seed varies BOTH weight init (--seed) and data order (--sampler_seed) for an independent draw.
#
# Usage: bash run_babylm_winner_seeds.sh <maxit> <warm> <batch> <gaccum>
#   e.g. bash run_babylm_winner_seeds.sh 4740 100 32 16
PY=/media/volume/yupei-data/envs/nanogpt/bin/python
MAXIT="${1:?max_iters}"; WARM="${2:?warmup_iters}"; B="${3:-32}"; GA="${4:-16}"
DONE="results/winnerseeds.done"; rm -f "$DONE"

run () {  # seed
  local S="$1" NAME="bl100m-d512L24-do0.1-s${1}"
  if grep -q "\"run_name\": \"${NAME}\"" results/experiments.jsonl 2>/dev/null; then
    echo "==== skip ${NAME} (already in experiments.jsonl) ===="; return; fi
  echo "================ ${NAME}  (seed=${S} sampler_seed=${S}) ================"
  "$PY" train.py config/train_babylm.py \
    --dataset=babylm_100m --max_iters="$MAXIT" --lr_decay_iters="$MAXIT" --warmup_iters="$WARM" --eval_interval=500 \
    --use_rmsnorm=True --use_swiglu=True --use_rope=True \
    --sampler=shuffle --dropout=0.1 \
    --n_embd=512 --n_layer=24 --n_head=8 \
    --batch_size="$B" --gradient_accumulation_steps="$GA" \
    --seed="$S" --sampler_seed="$S" \
    --wandb_run_name="$NAME" --out_dir="out-babylm/${NAME}" \
    || echo "!!! FAILED: ${NAME}"
}

run 1338
run 1339
echo "==== winner multi-seed complete (n=3 with existing seed 1337) ===="; touch "$DONE"
