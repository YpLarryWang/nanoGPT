#!/usr/bin/env bash
# GPT-BERT objective ratio study on the seeded 100M champion.
#
# Clean three-arm comparison: every arm uses the hybrid-born 16064 vocabulary and
# the same code path; only the number of causal microsteps changes.
#   hyb1of16  = reference GPT-BERT mix (1 causal : 15 masked)
#   hyb8of16  = balanced mix (8 causal : 8 masked)
#   hyb16of16 = all-causal control
#
# Primary downstream readout: BLiMP + reliable-4. avg5 is secondary because
# entity_tracking is noisy. Keep seed and sampler_seed paired at 1337.
#
# Usage: bash run_babylm_hybrid_100m.sh [max_iters=4740] [warmup=100] [batch=32] [gaccum=16]

set -euo pipefail

PY=/media/volume/yupei-data/envs/nanogpt/bin/python
MAXIT="${1:-4740}"
WARM="${2:-100}"
B="${3:-32}"
GA="${4:-16}"
DONE="results/hybrid_100m.done"

if [[ "$GA" -ne 16 ]]; then
  echo "error: this registered ratio study requires gradient_accumulation_steps=16" >&2
  exit 2
fi

rm -f "$DONE"

run () {
  local CAUSAL="$1"
  local NAME="bl100m-d512L32-do0.1-gate-hyb${CAUSAL}of16"
  local OUT="out-babylm/${NAME}"

  if grep -q "\"run_name\": \"${NAME}\"" results/experiments.jsonl 2>/dev/null; then
    echo "==== skip ${NAME} (already in experiments.jsonl) ===="
    return
  fi
  if [[ -e "$OUT" ]]; then
    echo "error: ${OUT} exists without a completed ledger row; inspect it before retrying" >&2
    return 1
  fi

  echo "================ ${NAME} ================"
  echo "start: $(date --iso-8601=seconds)"
  PYTHONUNBUFFERED=1 "$PY" train.py config/train_babylm.py \
    --dataset=babylm_100m \
    --max_iters="$MAXIT" --lr_decay_iters="$MAXIT" --warmup_iters="$WARM" --eval_interval=500 \
    --use_rmsnorm=True --use_swiglu=True --use_rope=True \
    --sampler=shuffle --sampler_seed=1337 --dropout=0.1 --use_attn_gate=True \
    --n_embd=512 --n_layer=32 --n_head=8 \
    --batch_size="$B" --gradient_accumulation_steps="$GA" \
    --seed=1337 --use_hybrid=True --causal_microsteps="$CAUSAL" \
    --wandb_log=True --wandb_run_name="$NAME" --out_dir="$OUT"
  echo "finish: $(date --iso-8601=seconds)"
}

run 1
run 8
run 16

echo "==== 100M hybrid ratio study complete ===="
touch "$DONE"
