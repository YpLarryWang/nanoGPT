#!/usr/bin/env bash
# Convert one final AttnRes checkpoint and run the three frozen Pick-B tasks.

set -euo pipefail

VARIANT="${1:?usage: diag_masked_zero_shot.sh <variant> <none|old|embed|random_count_matched>}"
MODE="${2:?usage: diag_masked_zero_shot.sh <variant> <none|old|embed|random_count_matched>}"
case "$MODE" in
  none|old|embed|random_count_matched) ;;
  *) echo "invalid mask mode: $MODE" >&2; exit 2 ;;
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
MASK_SEED=20260718

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

RESULT_NAME="${VARIANT}--mask${MODE}"
HFDIR="$HF_ROOT/$RESULT_NAME"
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

for task in blimp entity_tracking comps; do
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
    "$EVAL_PY" -m evaluation_pipeline.sentence_zero_shot.run \
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
  for task in blimp entity_tracking comps; do
    leaf="$(task_leaf "$task")"
    original="$EVAL_REPO/results/$VARIANT/main/zero_shot/causal/$task/$leaf"
    diagnostic="$EVAL_REPO/results/$RESULT_NAME/main/zero_shot/causal/$task/$leaf"
    for filename in best_temperature_report.txt predictions.json; do
      [[ -s "$original/$filename" ]] || {
        echo "missing legacy parity artifact: $original/$filename" >&2
        exit 1
      }
      cmp -s "$original/$filename" "$diagnostic/$filename" || {
        echo "none-mode parity failed: $task/$filename" >&2
        exit 1
      }
    done
  done
  echo ">> none-mode parity is byte-identical for BLiMP, COMPS, and entity tracking"
fi

echo ">> complete result=$EVAL_REPO/results/$RESULT_NAME mask_seed=$MASK_SEED"
