#!/usr/bin/env bash
# Evaluate one completed 100M official-dev run on one GPU without mutating CSVs.

set -euo pipefail

VARIANT="${VARIANT:?set VARIANT}"
GPU="${GPU:?set GPU}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA="${DATA:-/workspace}"
NANO_REPO="${NANO_REPO:-/workspace/repo/nanoGPT}"
EVAL_REPO="${EVAL_REPO:-/workspace/repo/babylm-eval/strict}"
EVAL_PY="${PY:-/workspace/envs/babylm-eval/bin/python}"
HF_ROOT="${HF_ROOT:-/workspace/hf-models}"
LOG_DIR="$NANO_REPO/logs/eval-offdev-100m"
LOG_PATH="$LOG_DIR/$VARIANT.log"
DONE_MARKER="$NANO_REPO/results/$VARIANT.causal-zero-shot.done"
FAILED_MARKER="$NANO_REPO/results/$VARIANT.causal-zero-shot.failed"

fail() {
  echo "FATAL: $*" >&2
  return 1
}

cleanup() {
  local status=$?
  if (( status != 0 )); then
    printf '%s status=%s log=%s\n' "$(date --iso-8601=seconds)" "$status" "$LOG_PATH" \
      > "$FAILED_MARKER"
  fi
}

trap cleanup EXIT
mkdir -p "$LOG_DIR" "$NANO_REPO/results" "$HF_ROOT"
exec > >(tee -a "$LOG_PATH") 2>&1

[[ -x "$EVAL_PY" ]] || fail "missing eval Python: $EVAL_PY"
[[ -d "$EVAL_REPO" ]] || fail "missing eval repo: $EVAL_REPO"
[[ -f "$NANO_REPO/out-babylm/$VARIANT/checkpoint_manifest.json" ]] || \
  fail "missing completed checkpoint manifest"
[[ ! -e "$DONE_MARKER" ]] || fail "evaluation already completed"
[[ ! -e "$FAILED_MARKER" ]] || fail "previous evaluation failure needs review"
[[ ! -e "$HF_ROOT/$VARIANT" ]] || fail "HF export already exists"
[[ ! -e "$EVAL_REPO/results/$VARIANT" ]] || fail "evaluation results already exist"

FINAL_CKPT="$($EVAL_PY -c '
import json, os, sys
p = sys.argv[1]
d = json.load(open(p))
f = os.path.join(os.path.dirname(p), d["roles"]["final"])
assert os.path.isfile(f), f
print(f)
' "$NANO_REPO/out-babylm/$VARIANT/checkpoint_manifest.json")"

USED="$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print int($1)}')"
(( USED <= 512 )) || fail "GPU $GPU is busy (${USED}MiB)"

echo "start=$(date --iso-8601=seconds) variant=$VARIANT gpu=$GPU checkpoint=$FINAL_CKPT"
CUDA_VISIBLE_DEVICES="$GPU" \
  DATA="$DATA" NANO_REPO="$NANO_REPO" EVAL_REPO="$EVAL_REPO" \
  PY="$EVAL_PY" HF_ROOT="$HF_ROOT" \
  bash eval/eval_variant.sh "$VARIANT" --no-sync

"$EVAL_PY" eval/sync_eval_results.py "$VARIANT" \
  --eval-repo "$EVAL_REPO" --backend causal --full \
  --metadata-from bl100m-d512L32-do0.1-gate --dry-run

printf '%s gpu=%s log=%s\n' "$(date --iso-8601=seconds)" "$GPU" "$LOG_PATH" > "$DONE_MARKER"
echo "complete=$(date --iso-8601=seconds) variant=$VARIANT gpu=$GPU"
