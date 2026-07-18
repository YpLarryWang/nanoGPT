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
WAIT_FOR_FINAL="${WAIT_FOR_FINAL:-0}"
POLL_SECONDS="${POLL_SECONDS:-30}"
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
[[ "$WAIT_FOR_FINAL" == 0 || "$WAIT_FOR_FINAL" == 1 ]] || fail "WAIT_FOR_FINAL must be 0 or 1"
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "POLL_SECONDS must be positive"
[[ ! -e "$DONE_MARKER" ]] || fail "evaluation already completed"
[[ ! -e "$FAILED_MARKER" ]] || fail "previous evaluation failure needs review"
[[ ! -e "$HF_ROOT/$VARIANT" ]] || fail "HF export already exists"
[[ ! -e "$EVAL_REPO/results/$VARIANT" ]] || fail "evaluation results already exist"

manifest_final() {
  "$EVAL_PY" -c '
import json, os, sys
p = sys.argv[1]
d = json.load(open(p))
name = d.get("roles", {}).get("final")
assert name, "final role is not recorded yet"
f = os.path.join(os.path.dirname(p), name)
assert os.path.isfile(f), f
print(f)
' "$NANO_REPO/out-babylm/$VARIANT/checkpoint_manifest.json"
}

MANIFEST="$NANO_REPO/out-babylm/$VARIANT/checkpoint_manifest.json"
while :; do
  if [[ -f "$MANIFEST" ]] && FINAL_CKPT="$(manifest_final 2>/dev/null)"; then
    break
  fi
  (( WAIT_FOR_FINAL == 1 )) || fail "missing completed final checkpoint"
  if ! pgrep -af '[p]ython.*train.py' 2>/dev/null | grep -F -- "--wandb_run_name=$VARIANT" >/dev/null; then
    fail "training is not active and no completed final checkpoint exists"
  fi
  echo "wait_final=$(date --iso-8601=seconds) variant=$VARIANT"
  sleep "$POLL_SECONDS"
done

while :; do
  USED="$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print int($1)}')"
  if (( USED <= 512 )); then
    break
  fi
  (( WAIT_FOR_FINAL == 1 )) || fail "GPU $GPU is busy (${USED}MiB)"
  echo "wait_gpu=$(date --iso-8601=seconds) gpu=$GPU memory_used=${USED}MiB"
  sleep "$POLL_SECONDS"
done

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
