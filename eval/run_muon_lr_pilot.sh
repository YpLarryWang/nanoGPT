#!/usr/bin/env bash
# AdamW fine-tuning LR pilot for the 10M Muon+AttnRes candidate.
#
# One invocation owns one GPU and one LR arm. The four arms are intended to run
# concurrently on four identical GPUs. Results are isolated from the official
# strict/results tree so the 3e-5 arm remains a pilot baseline, not a committed
# full-GLUE result.

set -euo pipefail

LR="${1:?usage: run_muon_lr_pilot.sh <learning-rate> [lr-tag]}"
LR_TAG="${2:-lr${LR}}"

: "${DATA:=/workspace}"
: "${NANO_REPO:=$DATA/nanoGPT}"
: "${EVAL_REPO:=$DATA/babylm-eval/strict}"
: "${PY:=$DATA/envs/babylm-eval/bin/python}"
: "${HF_ROOT:=$DATA/hf-models}"

VARIANT=muon-bl10m-d512L32-do0.1-gate-attnres8
HFDIR="$HF_ROOT/$VARIANT"
RESULTS_ROOT="results-lr-pilot/$VARIANT/$LR_TAG"
LOG_DIR="$NANO_REPO/logs/finetune-lr-pilot"
LOG_FILE="$LOG_DIR/$LR_TAG.log"

[[ -x "$PY" ]] || { echo "missing eval python: $PY" >&2; exit 1; }
[[ -f "$HFDIR/config.json" ]] || { echo "missing converted model: $HFDIR" >&2; exit 1; }

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

echo "pilot_start=$(date --iso-8601=seconds) variant=$VARIANT lr=$LR lr_tag=$LR_TAG gpu=${CUDA_VISIBLE_DEVICES:-unset} seed=42"
echo "fixed_config=adamw beta1=0.9 beta2=0.999 eps=1e-8 weight_decay=0.01 warmup=0.06 scheduler=cosine"
echo "tasks=mnli:e10:b32,qqp:e10:b16,rte:e10:b16,multirc:e5:b16"

run_task () {
  local TASK="$1" LABELS="$2" BSZ="$3" EPOCHS="$4" SELECT_METRIC="$5"
  shift 5
  local METRICS=("$@")
  local OUT="$RESULTS_ROOT/$VARIANT/main/finetune/$TASK"
  local EXP_NAME="${VARIANT}-ftadamw-${LR_TAG}-${TASK}-b${BSZ}-e${EPOCHS}-s42"

  if [[ -e "$OUT" ]]; then
    echo "error: existing pilot output: $OUT" >&2
    return 1
  fi

  echo "task_start=$(date --iso-8601=seconds) task=$TASK lr=$LR batch=$BSZ epochs=$EPOCHS"
  PYTHONUNBUFFERED=1 "$PY" -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$HFDIR" \
    --train_data "evaluation_data/full_eval/glue_filtered/$TASK.train.jsonl" \
    --valid_data "evaluation_data/full_eval/glue_filtered/$TASK.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/glue_filtered/$TASK.valid.jsonl" \
    --task "$TASK" \
    --num_labels "$LABELS" \
    --batch_size "$BSZ" \
    --learning_rate "$LR" \
    --num_epochs "$EPOCHS" \
    --sequence_length 512 \
    --results_dir "$RESULTS_ROOT" \
    --metrics "${METRICS[@]}" \
    --metric_for_valid "$SELECT_METRIC" \
    --seed 42 \
    --verbose \
    --padding_side left \
    --take_final \
    --wandb \
    --exp_name "$EXP_NAME"
  echo "task_finish=$(date --iso-8601=seconds) task=$TASK"
}

run_task mnli    3 32 10 accuracy accuracy
run_task qqp     2 16 10 f1       accuracy f1 mcc
run_task rte     2 16 10 accuracy accuracy f1 mcc
run_task multirc 2 16 5  accuracy accuracy f1 mcc

touch "$NANO_REPO/results/${VARIANT}.ftlrpilot.${LR_TAG}.done"
echo "pilot_finish=$(date --iso-8601=seconds) variant=$VARIANT lr=$LR"
