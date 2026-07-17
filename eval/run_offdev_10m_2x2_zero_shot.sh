#!/usr/bin/env bash
# Full causal zero-shot evaluation for the completed seed-1337 offdev 10M 2x2.
# Scoreboard import is deliberately deferred until all four raw result trees pass.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA=/media/volume/yupei-data
EVAL_REPO="$DATA/repo/babylm-eval/strict"
EVAL_PY="$DATA/envs/babylm-eval/bin/python"
LOG_DIR=logs/eval-offdev-10m-2x2
QUEUE_LOG="$LOG_DIR/causal-zero-shot.queue.log"
DONE_MARKER=results/offdev10m-2x2-s1337.causal-zero-shot.done
FAILED_MARKER=results/offdev10m-2x2-s1337.causal-zero-shot.failed
LOCK_DIR=/tmp/offdev10m-2x2-s1337-causal-zero-shot.lock

NAMES=(
  bl10m-d512L32-do0.1-gate-offdev-aoaw19-aoat20-u37-b8ga64
  bl10m-d512L32-do0.1-gate-attnres8-offdev-aoaw19-aoat20-u37-b8ga64
  bl10m-d512L16-do0.1-gate-offdev-aoaw19-aoat20-u37-b8ga64
  bl10m-d512L16-do0.1-gate-attnres4-offdev-aoaw19-aoat20-u37-b8ga64
)

fail() {
  echo "FATAL: $*" >&2
  return 1
}

[[ -n "${TMUX:-}" ]] || fail "launch this queue inside tmux"
[[ -x "$EVAL_PY" ]] || fail "missing eval Python: $EVAL_PY"
[[ -d "$EVAL_REPO" ]] || fail "missing eval repo: $EVAL_REPO"
[[ -f data/babylm_officialdev/tokenizer/bpe-16000.json ]] || fail "missing offdev tokenizer"
[[ ! -e "$QUEUE_LOG" ]] || fail "queue log already exists: $QUEUE_LOG"
[[ ! -e "$DONE_MARKER" ]] || fail "queue already completed: $DONE_MARKER"
[[ ! -e "$FAILED_MARKER" ]] || fail "previous queue failure needs review: $FAILED_MARKER"
mkdir "$LOCK_DIR" 2>/dev/null || fail "another causal zero-shot queue appears active"

cleanup() {
  local status=$?
  rmdir "$LOCK_DIR" 2>/dev/null || true
  if (( status != 0 )); then
    printf '%s status=%s log=%s\n' "$(date --iso-8601=seconds)" "$status" "$QUEUE_LOG" \
      > "$FAILED_MARKER"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$LOG_DIR" results
exec > >(tee -a "$QUEUE_LOG") 2>&1

echo "queue=offdev10m-2x2-s1337-causal-zero-shot"
echo "git_sha=$(git rev-parse HEAD)"
echo "eval_repo=$EVAL_REPO"
echo "tokenizer=data/babylm_officialdev/tokenizer/bpe-16000.json"

ACTIVE="$(pgrep -af '[e]val_variant|[e]val_zero_shot|evaluation_pipeline.*run' 2>/dev/null || true)"
[[ -z "$ACTIVE" ]] || { echo "$ACTIVE" >&2; fail "another evaluation is active"; }

for name in "${NAMES[@]}"; do
  manifest="out-babylm/$name/checkpoint_manifest.json"
  [[ -f "$manifest" ]] || fail "missing manifest: $manifest"
  "$EVAL_PY" -c 'import json,os,sys; p=sys.argv[1]; d=json.load(open(p)); f=os.path.join(os.path.dirname(p),d["roles"]["final"]); assert os.path.isfile(f),f; assert d["roles"]["best"]==d["roles"]["final"],d["roles"]' "$manifest"
  [[ ! -e "$DATA/hf-models/$name" ]] || fail "HF export already exists: $DATA/hf-models/$name"
  [[ ! -e "$EVAL_REPO/results/$name" ]] || fail "eval results already exist: $EVAL_REPO/results/$name"
  [[ ! -e "$LOG_DIR/$name.log" ]] || fail "eval log already exists: $LOG_DIR/$name.log"
done

for name in "${NAMES[@]}"; do
  echo "[$(date --iso-8601=seconds)] START full-causal-zero-shot:$name"
  set +e
  bash eval/eval_variant.sh "$name" --no-sync 2>&1 | tee "$LOG_DIR/$name.log"
  status=${PIPESTATUS[0]}
  set -e
  (( status == 0 )) || fail "$name evaluation exited with status $status"

  # Parse every required full result without writing scoreboards. This catches
  # official-pipeline task failures that otherwise sometimes return success.
  "$EVAL_PY" eval/sync_eval_results.py "$name" \
    --eval-repo "$EVAL_REPO" --backend causal --full \
    --metadata-from bl10m-d512L32-do0.1-gate --dry-run
  echo "[$(date --iso-8601=seconds)] COMPLETE full-causal-zero-shot:$name"
done

printf '%s sha=%s log=%s\n' "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$QUEUE_LOG" \
  > "$DONE_MARKER"
echo "==== offdev 10M seed-1337 2x2 causal zero-shot complete ===="
