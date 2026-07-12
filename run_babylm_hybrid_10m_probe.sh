#!/usr/bin/env bash
# Single-GPU 10M mostly-causal GPT-BERT probes.
# Usage: PY=/path/to/python CUDA_VISIBLE_DEVICES=N bash run_babylm_hybrid_10m_probe.sh {14|15}

set -euo pipefail

CAUSAL="${1:?usage: run_babylm_hybrid_10m_probe.sh 14-or-15}"
case "$CAUSAL" in
  14|15) ;;
  *) echo "expected 14 or 15 causal microsteps, got: $CAUSAL" >&2; exit 2 ;;
esac

PY="${PY:-python}"
NAME="bl10m-d512L32-do0.1-gate-hyb${CAUSAL}of16-b32ga16"
OUT="out-babylm/$NAME"

[[ ! -e "$OUT" ]] || { echo "output already exists: $OUT" >&2; exit 1; }
mkdir -p logs/hybrid

echo "run=$NAME dataset=babylm causal=$CAUSAL masked=$((16-CAUSAL)) batch=32 grad_accum=16 eval_batch=32 seed=1337"

exec "$PY" train.py config/train_babylm.py \
  --dataset=babylm \
  --max_iters=466 --lr_decay_iters=466 --warmup_iters=40 --eval_interval=50 \
  --use_rmsnorm=True --use_swiglu=True --use_rope=True --use_attn_gate=True \
  --use_attn_res=False --use_muon=False \
  --sampler=shuffle --sampler_seed=1337 --dropout=0.1 \
  --n_embd=512 --n_layer=32 --n_head=8 \
  --batch_size=32 --gradient_accumulation_steps=16 --eval_batch_size=32 \
  --seed=1337 --use_hybrid=True --causal_microsteps="$CAUSAL" \
  --wandb_log=True --wandb_project=babylm --wandb_run_name="$NAME" --out_dir="$OUT"
