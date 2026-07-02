#!/usr/bin/env bash
# Phase 2 -- multi-seed validation of the TOP-3 33M variants (entity_tracking has high seed variance:
# rms-swiglu-8/3 28.96 vs rms-swiglu-4 22.81 with everything else fixed). sampler='random' + batch 64/ga 8
# + dropout 0.0 -> identical to the original runs except the seed, so it combines with the existing
# seed-1337 numbers to give n=3 per variant. Base seed sets torch.manual_seed(seed) = init + data order.
#
# Usage: bash run_babylm_seeds.sh <maxit_100m> <warm_100m>
#   e.g. bash run_babylm_seeds.sh 4740 100
PY=/media/volume/yupei-data/envs/nanogpt/bin/python
MAXIT="${1:?max_iters for 100M}"; WARM="${2:?warmup_iters}"
DONE="results/seeds.done"; rm -f "$DONE"

run () {  # name rms swiglu rope seed
  local NAME="$1" RMS="$2" SW="$3" ROPE="$4" SEED="$5"
  if grep -q "\"run_name\": \"${NAME}\"" results/experiments.jsonl 2>/dev/null; then
    echo "==== skip ${NAME} (already in experiments.jsonl) ===="; return; fi
  echo "================ ${NAME}  (seed=${SEED}) ================"
  "$PY" train.py config/train_babylm.py \
    --dataset=babylm_100m --max_iters="$MAXIT" --lr_decay_iters="$MAXIT" --warmup_iters="$WARM" --eval_interval=500 \
    --use_rmsnorm="$RMS" --use_swiglu="$SW" --use_rope="$ROPE" \
    --seed="$SEED" \
    --wandb_run_name="$NAME" --out_dir="out-babylm/${NAME}" \
    || echo "!!! FAILED: ${NAME}"
}

for S in 1338 1339; do
  run "bl100m-rms-swiglu-rope-s${S}" True  True  True  "$S"   # #1 winner
  run "bl100m-rms-mlp-learned-s${S}" True  False False "$S"   # #2
  run "bl100m-ln-swiglu-rope-s${S}"  False True  True  "$S"   # #3
done
echo "==== multi-seed validation complete ===="; touch "$DONE"
