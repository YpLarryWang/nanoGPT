#!/usr/bin/env bash
# Phase 0 -- dropout validation for the WINNER arch (RMS+SwiGLU 8/3+RoPE), BEFORE any arch change.
# Everything except dropout matches the existing dropout=0.0 runs (sampler='random', batch 64/ga 8),
# so this is an apples-to-apples comparison against the numbers already in results/experiments.jsonl.
#   100M (strict):       dropout 0.1
#   10M  (strict-small): dropout 0.1, 0.2   (small data overfits more; guidance = 0.1 Strict / 0.2-0.3 Strict-small)
#
# Usage: bash run_babylm_dropout.sh <maxit_100m> <warm_100m>
#   e.g. bash run_babylm_dropout.sh 4740 100     # match maxit/warmup to the original bl100m runs
PY=/media/volume/yupei-data/envs/nanogpt/bin/python
MAXIT100="${1:?max_iters for 100M (match the original bl100m runs)}"; WARM100="${2:?warmup_iters for 100M}"
DONE="results/dropout.done"; rm -f "$DONE"   # completion marker (armed on start)

run () {  # name dataset maxit warm dropout evalint
  local NAME="$1" DS="$2" MAXIT="$3" WARM="$4" DO="$5" EVAL="$6"
  if grep -q "\"run_name\": \"${NAME}\"" results/experiments.jsonl 2>/dev/null; then
    echo "==== skip ${NAME} (already in experiments.jsonl) ===="; return; fi
  echo "================ ${NAME}  (dataset=${DS} dropout=${DO}) ================"
  "$PY" train.py config/train_babylm.py \
    --dataset="$DS" --max_iters="$MAXIT" --lr_decay_iters="$MAXIT" --warmup_iters="$WARM" --eval_interval="$EVAL" \
    --use_rmsnorm=True --use_swiglu=True --use_rope=True \
    --dropout="$DO" \
    --wandb_run_name="$NAME" --out_dir="out-babylm/${NAME}" \
    || echo "!!! FAILED: ${NAME}"
}

run bl100m-rms-swiglu-rope-do0.1 babylm_100m "$MAXIT100" "$WARM100" 0.1 500
run bl10m-rms-swiglu-rope-do0.1  babylm      466          40        0.1 50
run bl10m-rms-swiglu-rope-do0.2  babylm      466          40        0.2 50
echo "==== dropout validation complete ===="; touch "$DONE"
