#!/usr/bin/env bash
# Multi-seed the R2 WINNER: d512/L32 + dropout 0.1 + gated attention (~116M, "deep+dropout+gate").
# This is our best model so far (avg5 57.68, reliable-4 64.36, BLiMP 77.79 vs the 98M baseline's 74.73)
# but on a SINGLE seed (1337). Goal: error bars on that headline before it goes in the paper, and on
# the noisy entity_tracking task. Adds seeds 1338, 1339 -> n=3 with the existing seed-1337 run.
#
# Each seed varies BOTH weight init + dropout masks (--seed, global torch RNG) and data order
# (--sampler_seed, the shuffle schedule's private Generator) for a fully independent replicate.
# The two are decoupled in train.py, so we move them together to get a genuine draw.
#
# Usage: bash run_babylm_gate_seeds.sh <maxit> <warm> <batch> <gaccum>
#   e.g. bash run_babylm_gate_seeds.sh 4740 100 32 16   (drop to 16 32 if L32+gate OOMs; same tokens/iter)
PY=/media/volume/yupei-data/envs/nanogpt/bin/python
MAXIT="${1:?max_iters}"; WARM="${2:?warmup_iters}"; B="${3:-32}"; GA="${4:-16}"
DONE="results/gateseeds.done"; rm -f "$DONE"

run () {  # seed
  local S="$1" NAME="bl100m-d512L32-do0.1-gate-s${1}"
  if grep -q "\"run_name\": \"${NAME}\"" results/experiments.jsonl 2>/dev/null; then
    echo "==== skip ${NAME} (already in experiments.jsonl) ===="; return; fi
  echo "================ ${NAME}  (seed=${S} sampler_seed=${S} L=32 gate=True) ================"
  "$PY" train.py config/train_babylm.py \
    --dataset=babylm_100m --max_iters="$MAXIT" --lr_decay_iters="$MAXIT" --warmup_iters="$WARM" --eval_interval=500 \
    --use_rmsnorm=True --use_swiglu=True --use_rope=True \
    --sampler=shuffle --dropout=0.1 --use_attn_gate=True \
    --n_embd=512 --n_layer=32 --n_head=8 \
    --batch_size="$B" --gradient_accumulation_steps="$GA" \
    --seed="$S" --sampler_seed="$S" \
    --wandb_run_name="$NAME" --out_dir="out-babylm/${NAME}" \
    || echo "!!! FAILED: ${NAME}"
}

run 1338
run 1339
echo "==== gate multi-seed complete (n=3 with existing seed 1337) ===="; touch "$DONE"
