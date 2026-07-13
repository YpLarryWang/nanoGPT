#!/usr/bin/env bash
# GLUE-only evaluation for one converted nanoGPT variant.
#
# This mirrors babylm-eval/strict/scripts/eval_finetuning.sh, but is fail-fast
# and uses the known-safe 20GB MIG batch sizes for the 116M L32 model:
# BoolQ/MultiRC/RTE/QQP=16; WSC/MRPC/MNLI=32. It does not rerun zero-shot.
# Every task logs to the official "BabyLM Finetuning" W&B project.
#
# Usage: bash eval/eval_glue_variant.sh <variant>

set -euo pipefail

VARIANT="${1:?usage: eval_glue_variant.sh <variant>}"

: "${DATA:=/media/volume/yupei-data}"
: "${NANO_REPO:=$DATA/repo/nanoGPT}"
: "${EVAL_REPO:=$DATA/repo/babylm-eval/strict}"
: "${PY:=$DATA/envs/babylm-eval/bin/python}"
: "${HF_ROOT:=$DATA/hf-models}"

HFDIR="$HF_ROOT/$VARIANT"
[[ -f "$HFDIR/config.json" ]] || { echo "no converted model: $HFDIR" >&2; exit 1; }

export PATH="$(dirname "$PY"):$PATH"
cd "$EVAL_REPO"

if [[ -f ../.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source ../.env
  set +a
fi

run_task () {
  local TASK="$1" LABELS="$2" BSZ="$3" EPOCHS="$4" SELECT_METRIC="$5"
  shift 5
  local METRICS=("$@")
  local OUT="results/$VARIANT/main/finetune/$TASK"
  local SAVED="models/$VARIANT/$TASK"

  if [[ -e "$OUT" || -e "$SAVED" ]]; then
    echo "error: existing GLUE output for $VARIANT/$TASK; inspect before retrying" >&2
    return 1
  fi

  echo "================ $VARIANT / $TASK (bsz=$BSZ epochs=$EPOCHS) ================"
  echo "start: $(date --iso-8601=seconds)"
  PYTHONUNBUFFERED=1 "$PY" -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$HFDIR" \
    --train_data "evaluation_data/full_eval/glue_filtered/$TASK.train.jsonl" \
    --valid_data "evaluation_data/full_eval/glue_filtered/$TASK.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/glue_filtered/$TASK.valid.jsonl" \
    --task "$TASK" \
    --num_labels "$LABELS" \
    --batch_size "$BSZ" \
    --learning_rate 3e-5 \
    --num_epochs "$EPOCHS" \
    --sequence_length 512 \
    --results_dir results \
    --save --save_dir models \
    --metrics "${METRICS[@]}" \
    --metric_for_valid "$SELECT_METRIC" \
    --seed 42 \
    --verbose \
    --padding_side left \
    --take_final \
    --wandb
  echo "finish: $(date --iso-8601=seconds)"
}

run_task boolq   2 16 10 accuracy accuracy f1 mcc
run_task multirc 2 16 10 accuracy accuracy f1 mcc
run_task rte     2 16 10 accuracy accuracy f1 mcc
run_task wsc     2 32 30 accuracy accuracy f1 mcc
run_task mrpc    2 32 10 f1       accuracy f1 mcc
run_task qqp     2 16 10 f1       accuracy f1 mcc
run_task mnli    3 32 10 accuracy accuracy

touch "$NANO_REPO/results/${VARIANT}.glue.done"
SYNC_ARGS=("$VARIANT" --glue --eval-repo "$EVAL_REPO")
if [[ -n "${SYNC_METADATA_FROM:-}" ]]; then
  SYNC_ARGS+=(--metadata-from "$SYNC_METADATA_FROM")
fi
"$PY" "$NANO_REPO/eval/sync_eval_results.py" "${SYNC_ARGS[@]}"
echo "==== GLUE complete: $VARIANT ===="
