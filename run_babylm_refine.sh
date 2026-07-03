#!/usr/bin/env bash
# Refine the scaled winner (deep+dropout) before layering on architecture changes. Two questions:
#   (1) does dropout 0.1 also help the smaller 58M (which barely overfit at do0.0)?  -> d512/L16 + do0.1
#   (2) does depth keep paying PAST L24?                                             -> d512/L32 ~108M + do0.1
# Winner arch RMS+SwiGLU 8/3+RoPE, sampler=shuffle, dropout 0.1, 100M strict, 10 epochs.
# head_dim=64 (n_head=8). Compare against the existing d512L16 (do0.0, avg5 55.55) and
# d512L24+do0.1 (3-seed mean 55.28).
#
# Usage: bash run_babylm_refine.sh <maxit> <warm> <batch> <gaccum>
#   e.g. bash run_babylm_refine.sh 4740 100 32 16     (drop to 16 32 if L32 OOMs on the 20GB MIG)
PY=/media/volume/yupei-data/envs/nanogpt/bin/python
MAXIT="${1:?max_iters}"; WARM="${2:?warmup_iters}"; B="${3:-32}"; GA="${4:-16}"
DONE="results/refine.done"; rm -f "$DONE"

run () {  # name n_layer
  local NAME="$1" L="$2"
  if grep -q "\"run_name\": \"${NAME}\"" results/experiments.jsonl 2>/dev/null; then
    echo "==== skip ${NAME} (already in experiments.jsonl) ===="; return; fi
  echo "================ ${NAME}  (d=512 L=${L} do0.1 B=${B} ga=${GA}) ================"
  "$PY" train.py config/train_babylm.py \
    --dataset=babylm_100m --max_iters="$MAXIT" --lr_decay_iters="$MAXIT" --warmup_iters="$WARM" --eval_interval=500 \
    --use_rmsnorm=True --use_swiglu=True --use_rope=True \
    --sampler=shuffle --dropout=0.1 \
    --n_embd=512 --n_layer="$L" --n_head=8 \
    --batch_size="$B" --gradient_accumulation_steps="$GA" \
    --wandb_run_name="$NAME" --out_dir="out-babylm/${NAME}" \
    || echo "!!! FAILED: ${NAME}"
}

run bl100m-d512L16-do0.1 16
run bl100m-d512L32-do0.1 32
echo "==== refine complete ===="; touch "$DONE"
