#!/usr/bin/env bash
# R1 -- dropout sweep on the overfitting 108M (d512/L32). At dropout 0.1 it val-mins at ~8.4 epochs
# then rises (spare capacity overfitting the 100M corpus). Test whether MORE dropout flattens the
# curve AND pushes the val-minimum lower (a genuinely better model, since we keep the best-val ckpt).
# Winner arch RMS+SwiGLU 8/3+RoPE, sampler=shuffle, 100M strict, 10 epochs. head_dim=64.
# Compare against the existing bl100m-d512L32-do0.1 (best_val 3.1958, reliable-4 63.55).
#
# Usage: bash run_babylm_do_sweep.sh <maxit> <warm> <batch> <gaccum>
#   e.g. bash run_babylm_do_sweep.sh 4740 100 32 16
PY=/media/volume/yupei-data/envs/nanogpt/bin/python
MAXIT="${1:?max_iters}"; WARM="${2:?warmup_iters}"; B="${3:-32}"; GA="${4:-16}"
DONE="results/dosweep.done"; rm -f "$DONE"

run () {  # dropout
  local DO="$1" NAME="bl100m-d512L32-do${1}"
  if grep -q "\"run_name\": \"${NAME}\"" results/experiments.jsonl 2>/dev/null; then
    echo "==== skip ${NAME} (already in experiments.jsonl) ===="; return; fi
  echo "================ ${NAME}  (d=512 L=32 dropout=${DO} B=${B} ga=${GA}) ================"
  "$PY" train.py config/train_babylm.py \
    --dataset=babylm_100m --max_iters="$MAXIT" --lr_decay_iters="$MAXIT" --warmup_iters="$WARM" --eval_interval=500 \
    --use_rmsnorm=True --use_swiglu=True --use_rope=True \
    --sampler=shuffle --dropout="$DO" \
    --n_embd=512 --n_layer=32 --n_head=8 \
    --batch_size="$B" --gradient_accumulation_steps="$GA" \
    --wandb_run_name="$NAME" --out_dir="out-babylm/${NAME}" \
    || echo "!!! FAILED: ${NAME}"
}

run 0.15
run 0.2
echo "==== dropout sweep complete ===="; touch "$DONE"
