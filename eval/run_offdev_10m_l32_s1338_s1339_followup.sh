#!/usr/bin/env bash
# On a 4-GPU Vast host, evaluate each L32 run on its training GPU as soon as
# that run finishes. Workers are independent so a slower AttnRes run does not
# leave GPUs from completed baseline runs idle.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA="${DATA:-/workspace}"
NANO_REPO="${NANO_REPO:-$DATA/nanoGPT}"
EVAL_REPO="${EVAL_REPO:-$DATA/repo/babylm-eval/strict}"
EVAL_PY="${PY:-$DATA/envs/babylm-eval/bin/python}"
HF_ROOT="${HF_ROOT:-$DATA/hf-models}"
QUEUE_NAME=offdev10m-l32-s1338-s1339-zero-shot-followup
LOG_DIR=logs/eval-offdev-10m-l32-s1338-s1339
QUEUE_LOG="$LOG_DIR/${QUEUE_NAME}.queue.log"
DONE_MARKER=results/${QUEUE_NAME}.done
FAILED_MARKER=results/${QUEUE_NAME}.failed
LOCK_DIR=/tmp/${QUEUE_NAME}.lock
TRAIN_FAILED_MARKER=results/offdev10m-l32-s1338-s1339.queue.failed

GPUS=(0 1 2 3)
NAMES=(
  bl10m-d512L32-do0.1-gate-offdev-aoaw19-aoat20-u37-b8ga64-s1338
  bl10m-d512L32-do0.1-gate-attnres8-offdev-aoaw19-aoat20-u37-b8ga64-s1338
  bl10m-d512L32-do0.1-gate-offdev-aoaw19-aoat20-u37-b8ga64-s1339
  bl10m-d512L32-do0.1-gate-attnres8-offdev-aoaw19-aoat20-u37-b8ga64-s1339
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

training_is_active() {
  local name="$1"
  pgrep -af '[p]ython.*train.py' 2>/dev/null | \
    grep -F -- "--wandb_run_name=$name" >/dev/null
}

wait_for_final_checkpoint() {
  local name="$1"
  local manifest="out-babylm/$name/checkpoint_manifest.json"

  echo "[$(date --iso-8601=seconds)] WAIT training:$name"
  while training_is_active "$name"; do
    if [[ -e "$TRAIN_FAILED_MARKER" ]]; then
      fail "training queue failed while waiting for $name"
      return 1
    fi
    sleep 5
  done
  if [[ -e "$TRAIN_FAILED_MARKER" ]]; then
    fail "training queue failed before evaluating $name"
    return 1
  fi
  if [[ ! -f "$manifest" ]]; then
    fail "training exited without manifest: $manifest"
    return 1
  fi
  "$EVAL_PY" -c '
import json, os, sys
p = sys.argv[1]
d = json.load(open(p))
f = os.path.join(os.path.dirname(p), d["roles"]["final"])
assert os.path.isfile(f), f
print(f)
' "$manifest" || return 1
  echo "[$(date --iso-8601=seconds)] READY training:$name"
}

wait_for_idle_gpu() {
  local gpu="$1"
  local used
  while :; do
    used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{ print int($1) }')"
    if (( used <= 512 )); then
      echo "gpu=$gpu memory_used=${used}MiB ready_for_eval"
      return 0
    fi
    echo "gpu=$gpu memory_used=${used}MiB waiting_for_release"
    sleep 5
  done
}

evaluate_one() {
  local index="$1"
  local gpu="${GPUS[$index]}"
  local name="${NAMES[$index]}"
  local log_path="$LOG_DIR/$name.log"
  local worker_done="results/$name.causal-zero-shot.done"

  wait_for_final_checkpoint "$name" || return 1
  wait_for_idle_gpu "$gpu" || return 1
  echo "[$(date --iso-8601=seconds)] START full-causal-zero-shot:$name gpu=$gpu"

  CUDA_VISIBLE_DEVICES="$gpu" \
    DATA="$DATA" NANO_REPO="$NANO_REPO" EVAL_REPO="$EVAL_REPO" \
    PY="$EVAL_PY" HF_ROOT="$HF_ROOT" \
    bash eval/eval_variant.sh "$name" --no-sync > "$log_path" 2>&1
  local status=$?
  if (( status != 0 )); then
    fail "$name evaluation exited with status $status"
    return 1
  fi

  "$EVAL_PY" eval/sync_eval_results.py "$name" \
    --eval-repo "$EVAL_REPO" --backend causal --full \
    --metadata-from bl10m-d512L32-do0.1-gate --dry-run >> "$log_path" 2>&1 || return 1

  printf '%s gpu=%s log=%s\n' "$(date --iso-8601=seconds)" "$gpu" "$log_path" \
    > "$worker_done"
  echo "[$(date --iso-8601=seconds)] COMPLETE full-causal-zero-shot:$name gpu=$gpu"
}

worker_entry() {
  local index="$1"
  local name="${NAMES[$index]}"
  local log_path="$LOG_DIR/$name.log"
  local worker_failed="results/$name.causal-zero-shot.failed"
  local status

  # This function runs in a background subshell; only the parent owns the
  # queue-level lock and markers.
  trap - EXIT INT TERM
  set +e
  evaluate_one "$index"
  status=$?
  set -e
  if (( status != 0 )); then
    printf '%s status=%s log=%s\n' "$(date --iso-8601=seconds)" "$status" "$log_path" \
      > "$worker_failed"
  fi
  return "$status"
}

[[ -n "${TMUX:-}" ]] || fail "launch this follow-up queue inside tmux"
[[ -x "$EVAL_PY" ]] || fail "missing eval Python: $EVAL_PY"
[[ -d "$EVAL_REPO" ]] || fail "missing eval repo: $EVAL_REPO"
[[ -f data/babylm_officialdev/tokenizer/bpe-16000.json ]] || fail "missing offdev tokenizer"
[[ ! -e "$QUEUE_LOG" ]] || fail "queue log already exists: $QUEUE_LOG"
[[ ! -e "$DONE_MARKER" ]] || fail "follow-up queue already completed: $DONE_MARKER"
[[ ! -e "$FAILED_MARKER" ]] || fail "previous follow-up failure needs review: $FAILED_MARKER"
[[ ! -e "$TRAIN_FAILED_MARKER" ]] || fail "training queue has already failed"
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
echo "hf_root=$HF_ROOT"

for index in "${!NAMES[@]}"; do
  name="${NAMES[$index]}"
  [[ ! -e "$HF_ROOT/$name" ]] || fail "HF export already exists: $HF_ROOT/$name"
  [[ ! -e "$EVAL_REPO/results/$name" ]] || fail "eval results already exist: $EVAL_REPO/results/$name"
  [[ ! -e "$LOG_DIR/$name.log" ]] || fail "eval log already exists: $LOG_DIR/$name.log"
  [[ ! -e "results/$name.causal-zero-shot.done" ]] || fail "worker already completed: $name"
  [[ ! -e "results/$name.causal-zero-shot.failed" ]] || fail "previous worker failure needs review: $name"
done

for index in "${!NAMES[@]}"; do
  worker_entry "$index" &
  WORKER_PIDS+=("$!")
done

overall_status=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "$pid"; then
    overall_status=1
  fi
done
(( overall_status == 0 )) || fail "one or more zero-shot follow-up workers failed"

printf '%s sha=%s log=%s\n' "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$QUEUE_LOG" \
  > "$DONE_MARKER"
echo "==== seed-1338/1339 offdev L32 zero-shot follow-up complete ===="
