#!/usr/bin/env bash
# Run the official BabyLM GLUE fine-tuning recipe, including F1 checkpoint
# selection for MRPC/QQP. The final summary nevertheless uses the seven task
# accuracies only. Official effective batches are preserved while
# GLUE_MICROBATCH controls memory use on the 20GB MIG.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VARIANT="${1:?usage: run_offdev_glue_accuracy.sh <offdev-variant>}"
[[ "$VARIANT" == *offdev* ]] || { echo "error: expected an offdev variant" >&2; exit 2; }

DATA="${DATA:-/media/volume/yupei-data}"
EVAL_REPO="${EVAL_REPO:-$DATA/repo/babylm-eval/strict}"
EVAL_PY="${PY:-$DATA/envs/babylm-eval/bin/python}"
HF_ROOT="${HF_ROOT:-$DATA/hf-models}"
HF_DIR="$HF_ROOT/$VARIANT"
LOG_DIR="logs/glue-offdev-accuracy"
QUEUE_LOG="$LOG_DIR/$VARIANT.log"
DONE_MARKER="$REPO_ROOT/results/$VARIANT.glue-accuracy.done"
FAILED_MARKER="$REPO_ROOT/results/$VARIANT.glue-accuracy.failed"
LOCK_DIR="/tmp/$VARIANT.glue-accuracy.lock"

LR=3e-5
MAX_EPOCHS=10
WSC_EPOCHS=30
SEED=42
SEQUENCE_LENGTH=512
GLUE_MICROBATCH="${GLUE_MICROBATCH:-16}"

fail() {
  echo "FATAL: $*" >&2
  return 1
}

cleanup() {
  local status=$?
  rmdir "$LOCK_DIR" 2>/dev/null || true
  if (( status != 0 )); then
    printf '%s status=%s log=%s\n' "$(date --iso-8601=seconds)" "$status" "$QUEUE_LOG" \
      > "$FAILED_MARKER"
  fi
}

run_task() {
  local task="$1" labels="$2" effective_batch="$3" epochs="$4" selection_metric="$5"
  shift 5
  local metrics=("$@")
  local micro_batch="$GLUE_MICROBATCH"
  local grad_accum=$((effective_batch / micro_batch))

  echo "task_start=$(date --iso-8601=seconds) task=$task lr=$LR effective_batch=$effective_batch microbatch=$micro_batch grad_accum=$grad_accum epochs=$epochs selection_metric=$selection_metric reported_metrics=${metrics[*]}"
  PYTHONUNBUFFERED=1 "$EVAL_PY" -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$HF_DIR" \
    --train_data "evaluation_data/full_eval/glue_filtered/$task.train.jsonl" \
    --valid_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
    --task "$task" \
    --num_labels "$labels" \
    --batch_size "$effective_batch" \
    --gradient_accumulation "$grad_accum" \
    --learning_rate "$LR" \
    --num_epochs "$epochs" \
    --sequence_length "$SEQUENCE_LENGTH" \
    --results_dir results \
    --save --save_dir models \
    --metrics "${metrics[@]}" \
    --metric_for_valid "$selection_metric" \
    --seed "$SEED" \
    --verbose \
    --padding_side left \
    --take_final \
    --wandb
  grep -Eq '^accuracy:[[:space:]]*[01](\.[0-9]+)?$' \
    "results/$VARIANT/main/finetune/$task/results.txt" \
    || fail "missing valid accuracy result for $task"
  echo "task_finish=$(date --iso-8601=seconds) task=$task"
}

[[ -n "${TMUX:-}" ]] || fail "launch this queue inside tmux"
[[ -x "$EVAL_PY" ]] || fail "missing eval Python: $EVAL_PY"
[[ -d "$EVAL_REPO" ]] || fail "missing eval repo: $EVAL_REPO"
[[ -f "$HF_DIR/config.json" ]] || fail "missing converted model: $HF_DIR"
[[ "$GLUE_MICROBATCH" =~ ^[1-9][0-9]*$ ]] || fail "GLUE_MICROBATCH must be a positive integer"
(( 16 % GLUE_MICROBATCH == 0 && 32 % GLUE_MICROBATCH == 0 )) \
  || fail "GLUE_MICROBATCH must divide official effective batches 16 and 32"
[[ ! -e "$QUEUE_LOG" ]] || fail "queue log already exists: $QUEUE_LOG"
[[ ! -e "$DONE_MARKER" ]] || fail "queue already completed: $DONE_MARKER"
[[ ! -e "$FAILED_MARKER" ]] || fail "previous queue failure needs review: $FAILED_MARKER"
[[ ! -e "$EVAL_REPO/results/$VARIANT/main/finetune" ]] \
  || fail "existing finetune results need review"
[[ ! -e "$EVAL_REPO/models/$VARIANT" ]] || fail "existing saved FT models need review"
mkdir "$LOCK_DIR" 2>/dev/null || fail "another GLUE queue appears active for $VARIANT"

ACTIVE="$(pgrep -af '[e]valuation_pipeline.finetune.run|[e]val_glue' 2>/dev/null || true)"
[[ -z "$ACTIVE" ]] || { echo "$ACTIVE" >&2; fail "another GLUE evaluation is active"; }

mkdir -p "$LOG_DIR" results
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec > >(tee -a "$QUEUE_LOG") 2>&1

echo "queue=offdev-glue-all-accuracy"
echo "variant=$VARIANT"
echo "git_sha=$(git rev-parse HEAD)"
echo "eval_repo=$EVAL_REPO"
echo "protocol=official-default-ft-metrics-and-selection;final-summary=seven-task-accuracy-mean"
echo "microbatch=$GLUE_MICROBATCH;effective-batches=16,32"

export PATH="$(dirname "$EVAL_PY"):$PATH"
cd "$EVAL_REPO"
if [[ -f ../.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source ../.env
  set +a
fi

run_task boolq 2 16 "$MAX_EPOCHS" accuracy accuracy f1 mcc
run_task multirc 2 16 "$MAX_EPOCHS" accuracy accuracy f1 mcc
run_task rte 2 32 "$MAX_EPOCHS" accuracy accuracy f1 mcc
run_task wsc 2 32 "$WSC_EPOCHS" accuracy accuracy f1 mcc
run_task mrpc 2 32 "$MAX_EPOCHS" f1 accuracy f1 mcc
run_task qqp 2 32 "$MAX_EPOCHS" f1 accuracy f1 mcc
run_task mnli 3 32 "$MAX_EPOCHS" accuracy accuracy

SUMMARY="results/$VARIANT/main/finetune/accuracy_summary.json"
"$EVAL_PY" "$REPO_ROOT/eval/summarize_glue_accuracy.py" "$VARIANT" \
  --eval-repo "$EVAL_REPO" --write-json "$SUMMARY"

printf '%s sha=%s summary=%s\n' "$(date --iso-8601=seconds)" "$(git -C "$REPO_ROOT" rev-parse HEAD)" "$SUMMARY" \
  > "$DONE_MARKER"
echo "==== GLUE all-accuracy complete: $VARIANT ===="
