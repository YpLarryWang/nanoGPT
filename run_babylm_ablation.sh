#!/usr/bin/env bash
# 2x2x2 architecture ablation for BabyLM: norm x mlp x position.
#   norm     : ln (LayerNorm)      | rms  (RMSNorm)
#   mlp      : mlp (vanilla 4x)    | swiglu (SwiGLU 8/3)
#   pos      : learned (abs wpe)   | rope (RoPE)
# 8 runs, each logged to wandb + results/experiments.jsonl. Continues on failure.
#
# Usage: bash run_babylm_ablation.sh <dataset> <max_iters> <warmup_iters> <prefix> [eval_interval]
#   bash run_babylm_ablation.sh babylm      466  40  bl10m
#   bash run_babylm_ablation.sh babylm_100m 4740 100 bl100m 500
PY=/media/volume/yupei-data/envs/nanogpt/bin/python
DS="${1:?dataset}"; MAXIT="${2:?max_iters}"; WARM="${3:?warmup_iters}"; PREFIX="${4:?prefix}"; EVAL="${5:-50}"
DONE="results/${PREFIX}_ablation.done"; rm -f "$DONE"   # completion marker (armed on start)

for norm in ln rms; do
  for mlp in mlp swiglu; do
    for pos in learned rope; do
      RMS=False;  [ "$norm" = rms ]    && RMS=True
      SW=False;   [ "$mlp"  = swiglu ] && SW=True
      ROPE=False; [ "$pos"  = rope ]   && ROPE=True
      NAME="${PREFIX}-${norm}-${mlp}-${pos}"
      if grep -q "\"run_name\": \"${NAME}\"" results/experiments.jsonl 2>/dev/null; then
        echo "==== skip ${NAME} (already in experiments.jsonl) ===="
        continue
      fi
      echo "================ ${NAME}  (dataset=${DS}) ================"
      "$PY" train.py config/train_babylm.py \
        --dataset="$DS" --max_iters="$MAXIT" --lr_decay_iters="$MAXIT" --warmup_iters="$WARM" --eval_interval="$EVAL" \
        --wandb_run_name="$NAME" --out_dir="out-babylm/${NAME}" \
        --use_rmsnorm=$RMS --use_swiglu=$SW --use_rope=$ROPE \
        || echo "!!! FAILED: ${NAME}"
    done
  done
done
echo "==== ablation complete: ${PREFIX} (dataset=${DS}) ===="
touch "$DONE"   # completion marker for the waiter
