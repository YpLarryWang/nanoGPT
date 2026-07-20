#!/usr/bin/env bash
# Convert one final AttnRes checkpoint and run frozen masking-control tasks.

set -euo pipefail

USAGE='usage: diag_masked_zero_shot.sh <variant> <none|old|embed|random_count_matched> [mask_seed] [all|blimp]'
VARIANT="${1:?$USAGE}"
MODE="${2:?$USAGE}"
MASK_SEED_ARG="${3:-}"
TASK_SELECTOR="${4:-all}"
case "$MODE" in
  none|old|embed|random_count_matched) ;;
  *) echo "invalid mask mode: $MODE" >&2; exit 2 ;;
esac
case "$TASK_SELECTOR" in
  all|blimp) ;;
  *) echo "invalid task selector: $TASK_SELECTOR" >&2; exit 2 ;;
esac
case "$VARIANT" in
  *attnres*) ;;
  *) echo "Pick B requires an AttnRes run, got: $VARIANT" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${NANO_REPO:=$REPO_ROOT}"
: "${DATA:=/media/volume/yupei-data}"
: "${EVAL_REPO:=$DATA/repo/babylm-eval/strict}"
: "${EVAL_PY:=$DATA/envs/babylm-eval/bin/python}"
: "${HF_ROOT:=$DATA/hf-models-diag}"
: "${RESUME:=0}"
if [[ "$MODE" == random_count_matched ]]; then
  MASK_SEED="${MASK_SEED_ARG:-${MASK_SEED:-}}"
  [[ "$MASK_SEED" =~ ^[0-9]+$ ]] || {
    echo "random_count_matched requires an explicit numeric mask_seed" >&2
    exit 2
  }
else
  [[ -z "$MASK_SEED_ARG" ]] || {
    echo "mask_seed is only valid for random_count_matched" >&2
    exit 2
  }
  : "${MASK_SEED:=20260718}"
fi

[[ "$RESUME" == 0 || "$RESUME" == 1 ]] || { echo "RESUME must be 0 or 1" >&2; exit 2; }
[[ -x "$EVAL_PY" ]] || { echo "missing eval Python: $EVAL_PY" >&2; exit 1; }
[[ -d "$EVAL_REPO" ]] || { echo "missing eval repository: $EVAL_REPO" >&2; exit 1; }

RUN_DIR="$NANO_REPO/out-babylm/$VARIANT"
MANIFEST="$RUN_DIR/checkpoint_manifest.json"
[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }
CKPT="$($EVAL_PY -c '
import json, os, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
name = d.get("roles", {}).get("final")
assert name, "manifest has no final role"
print(os.path.join(os.path.dirname(p), name))
' "$MANIFEST")"
[[ -f "$CKPT" ]] || { echo "missing final checkpoint: $CKPT" >&2; exit 1; }

case "$VARIANT" in
  *bl10m-*offdev*) TOK="$NANO_REPO/data/babylm_officialdev/tokenizer/bpe-16000.json" ;;
  *) echo "diagnosis script only supports the frozen 10M offdev runs" >&2; exit 2 ;;
esac
[[ -f "$TOK" ]] || { echo "missing tokenizer: $TOK" >&2; exit 1; }

if [[ "$MODE" == random_count_matched && "$MASK_SEED" != 20260718 ]]; then
  RESULT_NAME="${VARIANT}--mask${MODE}-seed${MASK_SEED}"
else
  # The validated legacy random draw is intentionally left at its unsuffixed
  # identity and is explicitly interpreted as mask_seed=20260718 by the parser.
  RESULT_NAME="${VARIANT}--mask${MODE}"
fi
if [[ "$MODE" == random_count_matched ]]; then
  HFDIR="$HF_ROOT/${VARIANT}--maskrandom_count_matched"
else
  HFDIR="$HF_ROOT/$RESULT_NAME"
fi
if [[ ! -f "$HFDIR/checkpoint_source.json" ]]; then
  [[ ! -e "$HFDIR" ]] || { echo "incomplete HF export exists: $HFDIR" >&2; exit 1; }
  "$EVAL_PY" "$NANO_REPO/eval/convert_nanogpt_to_hf.py" \
    --ckpt "$CKPT" --tokenizer "$TOK" --out "$HFDIR" --dtype float32
else
  "$EVAL_PY" -c '
import json, os, sys
p, expected = sys.argv[1:]
d = json.load(open(p, encoding="utf-8"))
assert d.get("filename") == os.path.basename(expected), (d.get("filename"), expected)
' "$HFDIR/checkpoint_source.json" "$CKPT"
  cmp -s "$NANO_REPO/eval/hf_nanogpt/modeling_nanogpt.py" "$HFDIR/modeling_nanogpt.py" || {
    echo "cached HF export has stale modeling_nanogpt.py: $HFDIR" >&2
    exit 1
  }
fi

export PATH="$(dirname "$EVAL_PY"):$PATH"
export PYTHONUNBUFFERED=1
export NANOGPT_ATTNRES_MASK="$MODE"
export NANOGPT_ATTNRES_MASK_SEED="$MASK_SEED"
export NANOGPT_EVAL_SEED="$MASK_SEED"
export PYTHONHASHSEED="$MASK_SEED"

task_leaf() {
  case "$1" in
    blimp) echo blimp_filtered ;;
    entity_tracking|comps) echo "$1" ;;
    *) return 2 ;;
  esac
}

task_complete() {
  local task="$1" leaf root
  leaf="$(task_leaf "$task")"
  root="$EVAL_REPO/results/$RESULT_NAME/main/zero_shot/causal/$task/$leaf"
  [[ -s "$root/best_temperature_report.txt" && -s "$root/predictions.json" ]]
}

report_summary() {
  "$EVAL_PY" -c '
import pathlib, sys
lines = pathlib.Path(sys.argv[1]).read_text().splitlines()
temperature = next(x for x in lines if x.startswith("TEMPERATURE:"))
i = next(i for i, x in enumerate(lines) if x.startswith("### AVERAGE "))
average = next(x for x in lines[i + 1:] if x.strip())
print(temperature + "\n" + average)
' "$1"
}

TASKS=(blimp entity_tracking comps)
if [[ "$TASK_SELECTOR" == blimp ]]; then TASKS=(blimp); fi
if [[ "$MODE" == random_count_matched && "$MASK_SEED" != 20260718 && "$TASK_SELECTOR" != blimp ]]; then
  echo "supplementary random draws are BLiMP-only; pass task selector 'blimp'" >&2
  exit 2
fi

for task in "${TASKS[@]}"; do
  if task_complete "$task"; then
    if [[ "$RESUME" == 1 ]]; then
      echo ">> $task complete; skipping"
      continue
    fi
    echo "result already exists for $task; set RESUME=1 to reuse" >&2
    exit 1
  fi
  TASK_ROOT="$EVAL_REPO/results/$RESULT_NAME/main/zero_shot/causal/$task"
  [[ ! -e "$TASK_ROOT" ]] || { echo "incomplete task output exists: $TASK_ROOT" >&2; exit 1; }
  echo ">> mask=$MODE task=$task model=$RESULT_NAME"
  DATASET="$task"
  if [[ "$task" == blimp ]]; then DATASET=blimp_filtered; fi
  (
    cd "$EVAL_REPO"
    "$EVAL_PY" -c '
import os
import runpy
import torch

torch.manual_seed(int(os.environ["NANOGPT_EVAL_SEED"]))
runpy.run_module("evaluation_pipeline.sentence_zero_shot.run", run_name="__main__")
' \
      --model_path_or_name "$HFDIR" \
      --backend causal \
      --task "$task" \
      --data_path "evaluation_data/full_eval/$DATASET" \
      --save_predictions \
      --revision_name main
  )
  task_complete "$task" || { echo "$task did not produce report + predictions" >&2; exit 1; }
done

if [[ "$MODE" == none ]]; then
  [[ "$TASK_SELECTOR" == all ]] || {
    echo "none parity requires task selector 'all'" >&2
    exit 2
  }
  for task in "${TASKS[@]}"; do
    leaf="$(task_leaf "$task")"
    original="$EVAL_REPO/results/$VARIANT/main/zero_shot/causal/$task/$leaf"
    diagnostic="$EVAL_REPO/results/$RESULT_NAME/main/zero_shot/causal/$task/$leaf"
    for filename in best_temperature_report.txt predictions.json; do
      [[ -s "$original/$filename" ]] || {
        echo "missing legacy parity artifact: $original/$filename" >&2
        exit 1
      }
    done
    cmp -s "$original/predictions.json" "$diagnostic/predictions.json" || {
      echo "none-mode parity failed: $task/predictions.json" >&2
      exit 1
    }
    if ! cmp -s "$original/best_temperature_report.txt" "$diagnostic/best_temperature_report.txt"; then
      original_summary="$(report_summary "$original/best_temperature_report.txt")"
      diagnostic_summary="$(report_summary "$diagnostic/best_temperature_report.txt")"
      [[ "$original_summary" == "$diagnostic_summary" ]] || {
        echo "none-mode parity failed: $task report headline" >&2
        exit 1
      }
      echo ">> warning: $task granular report differs despite byte-identical predictions; evaluator tie breaking can change candidate-index accuracy"
    fi
  done
  echo ">> none-mode parity passed: byte-identical predictions and identical report headlines for all three tasks"
fi

echo ">> complete result=$EVAL_REPO/results/$RESULT_NAME mask_seed=$MASK_SEED"
