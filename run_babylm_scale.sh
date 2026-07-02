#!/usr/bin/env bash
# Phase 1 -- scale-up ladder (depth-first), WINNER arch (RMS+SwiGLU 8/3+RoPE), 100M strict, sampler=shuffle.
#   d512/L16 ~58M | d512/L24 ~83M (primary) | d768/L10 ~83M (width control) | d768/L12 ~97M (GPT-2 shape @16k vocab)
# head_dim held at 64 (n_head = n_embd/64). batch_size/grad_accum passed in so we can hold
# tokens/iter = 262,144 (== the 33M runs) while shrinking the micro-batch to fit the 20GB MIG;
# grad-accum makes any B x GA with B*GA*block==262144 mathematically identical (LayerNorm/RMSNorm are per-sample).
#
# Usage: bash run_babylm_scale.sh <maxit> <warm> <batch> <gaccum>
#   e.g. bash run_babylm_scale.sh 4740 100 32 16     # 32*512*16 = 262,144 tok/iter
PY=/media/volume/yupei-data/envs/nanogpt/bin/python
MAXIT="${1:?max_iters}"; WARM="${2:?warmup_iters}"; B="${3:-32}"; GA="${4:-16}"
DONE="results/scale.done"; rm -f "$DONE"

run () {  # name n_embd n_layer n_head
  local NAME="$1" D="$2" L="$3" H="$4"
  if grep -q "\"run_name\": \"${NAME}\"" results/experiments.jsonl 2>/dev/null; then
    echo "==== skip ${NAME} (already in experiments.jsonl) ===="; return; fi
  echo "================ ${NAME}  (d=${D} L=${L} H=${H} B=${B} ga=${GA}) ================"
  "$PY" train.py config/train_babylm.py \
    --dataset=babylm_100m --max_iters="$MAXIT" --lr_decay_iters="$MAXIT" --warmup_iters="$WARM" --eval_interval=500 \
    --use_rmsnorm=True --use_swiglu=True --use_rope=True \
    --sampler=shuffle \
    --n_embd="$D" --n_layer="$L" --n_head="$H" \
    --batch_size="$B" --gradient_accumulation_steps="$GA" \
    --wandb_run_name="$NAME" --out_dir="out-babylm/${NAME}" \
    || echo "!!! FAILED: ${NAME}"
}

run bl100m-d512L16 512 16 8
run bl100m-d512L24 512 24 8
run bl100m-d768L10 768 10 12
run bl100m-d768L12 768 12 12
echo "==== scale-up ladder complete ===="; touch "$DONE"
