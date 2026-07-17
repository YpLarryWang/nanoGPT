#!/usr/bin/env bash
# Evaluate the completed offdev 10M L16 seed-1339 baseline/AttnRes4 pair
# concurrently on the JetStream A100 MIG. Each worker has isolated HF, result,
# log, and marker paths; scoreboard synchronization is deliberately deferred.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA="${DATA:-/media/volume/yupei-data}"
NANO_REPO="${NANO_REPO:-$DATA/repo/nanoGPT}"
EVAL_REPO="${EVAL_REPO:-$DATA/repo/babylm-eval/strict}"
EVAL_PY="${PY:-$DATA/envs/babylm-eval/bin/python}"
HF_ROOT="${HF_ROOT:-$DATA/hf-models}"
QUEUE_NAME=offdev10m-l16-s1339-zero-shot
LOG_DIR=logs/eval-offdev-10m-l16-s1339
QUEUE_LOG="$LOG_DIR/${QUEUE_NAME}.queue.log"
DONE_MARKER="results/${QUEUE_NAME}.done"
FAILED_MARKER="results/${QUEUE_NAME}.failed"
LOCK_DIR="/tmp/${QUEUE_NAME}.lock"
TRAIN_FAILED_MARKER=results/offdev10m-l16-s1339.queue.failed

NAMES=(
  bl10m-d512L16-do0.1-gate-offdev-aoaw19-aoat20-u37-b8ga64-s1339
  bl10m-d512L16-do0.1-gate-attnres4-offdev-aoaw19-aoat20-u37-b8ga64-s1339
)

WORKER_PIDS=()

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

validate_final_checkpoint() {
  local name="$1"
  local manifest="out-babylm/$name/checkpoint_manifest.json"

  [[ -f "$manifest" ]] || fail "missing manifest: $manifest"
  "$EVAL_PY" -c '
import json, os, sys
p = sys.argv[1]
d = json.load(open(p))
f = os.path.join(os.path.dirname(p), d["roles"]["final"])
assert os.path.isfile(f), f
assert os.path.getsize(f) > 0, f
print(f)
' "$manifest"
}

evaluate_one() {
  local name="$1"
  local log_path="$LOG_DIR/$name.log"
  local worker_done="results/$name.causal-zero-shot.done"

  echo "[$(date --iso-8601=seconds)] START full-causal-zero-shot:$name gpu=0"
  CUDA_VISIBLE_DEVICES=0 \
    DATA="$DATA" NANO_REPO="$NANO_REPO" EVAL_REPO="$EVAL_REPO" \
    PY="$EVAL_PY" HF_ROOT="$HF_ROOT" \
    bash eval/eval_variant.sh "$name" --no-sync > "$log_path" 2>&1 || return 1

  "$EVAL_PY" eval/sync_eval_results.py "$name" \
    --eval-repo "$EVAL_REPO" --backend causal --full \
    --metadata-from bl10m-d512L32-do0.1-gate --dry-run >> "$log_path" 2>&1 || return 1

  printf '%s gpu=0 log=%s\n' "$(date --iso-8601=seconds)" "$log_path" > "$worker_done"
  echo "[$(date --iso-8601=seconds)] COMPLETE full-causal-zero-shot:$name gpu=0"
}

worker_entry() {
  local name="$1"
  local log_path="$LOG_DIR/$name.log"
  local worker_failed="results/$name.causal-zero-shot.failed"
  local status

  trap - EXIT INT TERM
  set +e
  evaluate_one "$name"
  status=$?
  set -e
  if (( status != 0 )); then
    printf '%s status=%s log=%s\n' "$(date --iso-8601=seconds)" "$status" "$log_path" \
      > "$worker_failed"
  fi
  return "$status"
}

[[ -n "${TMUX:-}" ]] || fail "launch this queue inside tmux"
[[ -x "$EVAL_PY" ]] || fail "missing eval Python: $EVAL_PY"
[[ -d "$EVAL_REPO" ]] || fail "missing eval repo: $EVAL_REPO"
[[ -f data/babylm_officialdev/tokenizer/bpe-16000.json ]] || fail "missing offdev tokenizer"
[[ ! -e "$QUEUE_LOG" ]] || fail "queue log already exists: $QUEUE_LOG"
[[ ! -e "$DONE_MARKER" ]] || fail "queue already completed: $DONE_MARKER"
[[ ! -e "$FAILED_MARKER" ]] || fail "previous queue failure needs review: $FAILED_MARKER"
[[ ! -e "$TRAIN_FAILED_MARKER" ]] || fail "training queue has failed"
mkdir "$LOCK_DIR" 2>/dev/null || fail "another $QUEUE_NAME queue appears active"

mkdir -p "$LOG_DIR" results "$HF_ROOT"
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec > >(tee -a "$QUEUE_LOG") 2>&1

echo "queue=$QUEUE_NAME"
echo "git_sha=$(git rev-parse HEAD)"
echo "eval_repo=$EVAL_REPO"
echo "eval_python=$EVAL_PY"
echo "mode=two-concurrent-workers-on-one-MIG"

ACTIVE="$(pgrep -af '[e]val_variant|[e]val_zero_shot|evaluation_pipeline.*run' 2>/dev/null || true)"
[[ -z "$ACTIVE" ]] || { echo "$ACTIVE" >&2; fail "another evaluation is active"; }

for name in "${NAMES[@]}"; do
  validate_final_checkpoint "$name"
  [[ ! -e "$HF_ROOT/$name" ]] || fail "HF export already exists: $HF_ROOT/$name"
  [[ ! -e "$EVAL_REPO/results/$name" ]] || fail "eval results already exist: $EVAL_REPO/results/$name"
  [[ ! -e "$LOG_DIR/$name.log" ]] || fail "eval log already exists: $LOG_DIR/$name.log"
  [[ ! -e "results/$name.causal-zero-shot.done" ]] || fail "worker already completed: $name"
  [[ ! -e "results/$name.causal-zero-shot.failed" ]] || fail "previous worker failure needs review: $name"
done

for name in "${NAMES[@]}"; do
  worker_entry "$name" &
  WORKER_PIDS+=("$!")
done

overall_status=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "$pid"; then
    overall_status=1
  fi
done
(( overall_status == 0 )) || fail "one or more zero-shot workers failed"

printf '%s sha=%s log=%s\n' "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$QUEUE_LOG" \
  > "$DONE_MARKER"
echo "==== offdev 10M L16 seed-1339 causal zero-shot complete ===="
