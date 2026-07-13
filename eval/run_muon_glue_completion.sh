#!/usr/bin/env bash
# Complete the LR=1e-4 GLUE run for the 10M Muon+AttnRes candidate.
# One invocation owns one GPU and one task; supervisor launches four tasks in parallel.

set -euo pipefail

TASK="${1:?usage: run_muon_glue_completion.sh <boolq|wsc|mrpc|multirc>}"

: "${DATA:=/workspace}"
: "${NANO_REPO:=$DATA/nanoGPT}"
: "${EVAL_REPO:=$DATA/babylm-eval/strict}"
: "${PY:=$DATA/envs/babylm-eval/bin/python}"
: "${HF_ROOT:=$DATA/hf-models}"

VARIANT=muon-bl10m-d512L32-do0.1-gate-attnres8
HFDIR="$HF_ROOT/$VARIANT"
LR=1e-4
SEED=42
RESULTS_ROOT="results-lr-pilot/$VARIANT/lr1e-4-full7-completion"
LOG_DIR="$NANO_REPO/logs/finetune-lr-pilot"
LOG_FILE="$LOG_DIR/lr1e-4-full7-${TASK}.log"

case "$TASK" in
  boolq)   LABELS=2; EFFECTIVE_BSZ=16; GA=2; EPOCHS=10; SELECT_METRIC=accuracy; METRICS=(accuracy f1 mcc) ;;
  wsc)     LABELS=2; EFFECTIVE_BSZ=32; GA=4; EPOCHS=30; SELECT_METRIC=accuracy; METRICS=(accuracy f1 mcc) ;;
  mrpc)    LABELS=2; EFFECTIVE_BSZ=32; GA=4; EPOCHS=10; SELECT_METRIC=f1;       METRICS=(accuracy f1 mcc) ;;
  multirc) LABELS=2; EFFECTIVE_BSZ=16; GA=2; EPOCHS=10; SELECT_METRIC=accuracy; METRICS=(accuracy f1 mcc) ;;
  *) echo "unknown task: $TASK" >&2; exit 2 ;;
esac

MICRO_BSZ=$((EFFECTIVE_BSZ / GA))
OUT="$EVAL_REPO/$RESULTS_ROOT/$VARIANT/main/finetune/$TASK"
EXP_NAME="${VARIANT}-ftadamw-lr1e-4-${TASK}-eb${EFFECTIVE_BSZ}-mb${MICRO_BSZ}-ga${GA}-e${EPOCHS}-s${SEED}"

[[ -x "$PY" ]] || { echo "missing eval python: $PY" >&2; exit 1; }
[[ -f "$HFDIR/config.json" ]] || { echo "missing converted model: $HFDIR" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "existing completion output: $OUT" >&2; exit 1; }

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
export PATH="$(dirname "$PY"):$PATH"

cd "$EVAL_REPO"
if [[ -f ../.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source ../.env
  set +a
fi

echo "task_start=$(date --iso-8601=seconds) variant=$VARIANT task=$TASK lr=$LR seed=$SEED gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "effective_batch=$EFFECTIVE_BSZ microbatch=$MICRO_BSZ gradient_accumulation=$GA epochs=$EPOCHS select_metric=$SELECT_METRIC"

PYTHONUNBUFFERED=1 "$PY" -m evaluation_pipeline.finetune.run \
  --model_name_or_path "$HFDIR" \
  --train_data "evaluation_data/full_eval/glue_filtered/$TASK.train.jsonl" \
  --valid_data "evaluation_data/full_eval/glue_filtered/$TASK.valid.jsonl" \
  --predict_data "evaluation_data/full_eval/glue_filtered/$TASK.valid.jsonl" \
  --task "$TASK" \
  --num_labels "$LABELS" \
  --batch_size "$EFFECTIVE_BSZ" \
  --gradient_accumulation "$GA" \
  --learning_rate "$LR" \
  --num_epochs "$EPOCHS" \
  --sequence_length 512 \
  --results_dir "$RESULTS_ROOT" \
  --metrics "${METRICS[@]}" \
  --metric_for_valid "$SELECT_METRIC" \
  --seed "$SEED" \
  --verbose \
  --padding_side left \
  --take_final \
  --wandb \
  --exp_name "$EXP_NAME"

touch "$NANO_REPO/results/${VARIANT}.ftlr1e-4-full7.${TASK}.done"
echo "task_finish=$(date --iso-8601=seconds) variant=$VARIANT task=$TASK lr=$LR"
