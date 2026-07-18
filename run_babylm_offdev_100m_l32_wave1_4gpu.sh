#!/usr/bin/env bash
# Run the first four matched 100M official-dev L32 baseline/AttnRes8 jobs.
# Seeds 1337 and 1338 each occupy two GPUs; every run uses B16/GA32.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PY="${PY:-/workspace/envs/nanogpt/bin/python}"
DATASET=babylm_100m_officialdev
B=16
GA=32
MAX_ITERS=4797
STAGGER_SECONDS="${STAGGER_SECONDS:-45}"
POLL_SECONDS=15
MIN_START_FREE_GB=150
MIN_RUNTIME_FREE_GB=30
QUEUE_NAME=offdev100m-l32-wave1
LOG_DIR=logs/offdev-100m-l32-wave1
QUEUE_LOG="$LOG_DIR/${QUEUE_NAME}.queue.log"
DONE_MARKER=results/${QUEUE_NAME}.queue.done
FAILED_MARKER=results/${QUEUE_NAME}.queue.failed
LOCK_DIR=/tmp/${QUEUE_NAME}.lock

GPUS=(0 1 2 3)
SEEDS=(1337 1337 1338 1338)
ATTN_RES_BLOCKS=(0 8 0 8)
SCHEDULES=(
  config/checkpoint_schedules/bl100m-offdev-b16ga32-dual.json
  config/checkpoint_schedules/bl100m-offdev-b16ga32-dual.json
  config/checkpoint_schedules/bl100m-offdev-b16ga32-s1338-dual.json
  config/checkpoint_schedules/bl100m-offdev-b16ga32-s1338-dual.json
)
NAMES=(
  bl100m-d512L32-do0.1-gate-offdev-aoaw28-aoat31-u57-b16ga32
  bl100m-d512L32-do0.1-gate-attnres8-offdev-aoaw28-aoat31-u57-b16ga32
  bl100m-d512L32-do0.1-gate-offdev-aoaw28-aoat31-u57-b16ga32-s1338
  bl100m-d512L32-do0.1-gate-attnres8-offdev-aoaw28-aoat31-u57-b16ga32-s1338
)

PIDS=("" "" "" "")
ACTIVE=(0 0 0 0)
FATAL_PATTERN='CUDA out of memory|OutOfMemoryError|checkpoint schedule .*does not match|Traceback|(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)'

fail() {
  echo "FATAL: $*" >&2
  return 1
}

free_gb() {
  df -Pk "$REPO_ROOT" | awk 'NR == 2 { print int($4 / 1024 / 1024) }'
}

check_log() {
  local path="$1"
  [[ -f "$path" ]] || return 0
  if grep -Eiq "$FATAL_PATTERN" "$path"; then
    grep -Ein "$FATAL_PATTERN" "$path" | tail -20 >&2
    return 1
  fi
}

stop_active_runs() {
  local index pid
  for index in "${!PIDS[@]}"; do
    pid="${PIDS[$index]}"
    if [[ "${ACTIVE[$index]}" == 1 && -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for index in "${!PIDS[@]}"; do
    pid="${PIDS[$index]}"
    if [[ "${ACTIVE[$index]}" == 1 && -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
      ACTIVE[$index]=0
    fi
  done
}

cleanup() {
  local status=$?
  if (( status != 0 )); then
    stop_active_runs
    printf '%s status=%s log=%s\n' "$(date --iso-8601=seconds)" "$status" "$QUEUE_LOG" \
      > "$FAILED_MARKER"
  fi
  rmdir "$LOCK_DIR" 2>/dev/null || true
}

arch_args() {
  local attn_res_block="$1"
  ARCH_ARGS=(--n_layer=32 --n_embd=512 --n_head=8)
  if (( attn_res_block == 0 )); then
    ARCH_ARGS+=(--use_attn_res=False)
  else
    ARCH_ARGS+=(--use_attn_res=True --attn_res_block_size="$attn_res_block")
  fi
}

launch_run() {
  local index="$1"
  local gpu="${GPUS[$index]}"
  local seed="${SEEDS[$index]}"
  local name="${NAMES[$index]}"
  local schedule="${SCHEDULES[$index]}"
  local run_log="$LOG_DIR/$name.train.log"
  arch_args "${ATTN_RES_BLOCKS[$index]}"

  echo "[$(date --iso-8601=seconds)] START name=$name gpu=$gpu seed=$seed"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py config/train_babylm.py \
    --dataset="$DATASET" \
    --max_iters="$MAX_ITERS" --lr_decay_iters="$MAX_ITERS" --warmup_iters=100 \
    --eval_interval=500 --eval_iters=50 --eval_batch_size=32 \
    --checkpoint_schedule="$schedule" \
    --use_rmsnorm=True --use_swiglu=True --swiglu_mult=2.6666666666666665 \
    --use_rope=True --use_attn_gate=True \
    --use_muon=False --use_hybrid=False \
    --sampler=shuffle --sampler_seed="$seed" --dropout=0.1 \
    "${ARCH_ARGS[@]}" \
    --batch_size="$B" --gradient_accumulation_steps="$GA" \
    --seed="$seed" \
    --wandb_log=True --wandb_project=babylm \
    --wandb_run_name="$name" --out_dir="out-babylm/$name" > "$run_log" 2>&1 &
  PIDS[$index]=$!
  ACTIVE[$index]=1
  echo "launched pid=${PIDS[$index]} name=$name"
}

monitor_once() {
  local available index pid status run_log
  available="$(free_gb)"
  (( available >= MIN_RUNTIME_FREE_GB )) || fail \
    "disk below runtime floor: ${available}GB < ${MIN_RUNTIME_FREE_GB}GB"
  for index in "${!PIDS[@]}"; do
    [[ "${ACTIVE[$index]}" == 1 ]] || continue
    pid="${PIDS[$index]}"
    run_log="$LOG_DIR/${NAMES[$index]}.train.log"
    check_log "$run_log" || fail "fatal marker in ${NAMES[$index]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      set +e
      wait "$pid"
      status=$?
      set -e
      (( status == 0 )) || fail "status $status from ${NAMES[$index]}"
      check_log "$run_log" || fail "fatal marker after ${NAMES[$index]} completed"
      ACTIVE[$index]=0
      echo "[$(date --iso-8601=seconds)] COMPLETE name=${NAMES[$index]} gpu=${GPUS[$index]}"
    fi
  done
}

any_active() {
  local value
  for value in "${ACTIVE[@]}"; do
    [[ "$value" == 1 ]] && return 0
  done
  return 1
}

[[ -x "$PY" ]] || fail "missing training Python: $PY"
[[ "$STAGGER_SECONDS" =~ ^[0-9]+$ ]] || fail "invalid STAGGER_SECONDS"
[[ ! -e "$QUEUE_LOG" ]] || fail "queue log exists: $QUEUE_LOG"
[[ ! -e "$DONE_MARKER" ]] || fail "queue already completed"
[[ ! -e "$FAILED_MARKER" ]] || fail "previous failure marker needs review"
mkdir "$LOCK_DIR" 2>/dev/null || fail "queue lock exists: $LOCK_DIR"
mkdir -p "$LOG_DIR" results
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec > >(tee -a "$QUEUE_LOG") 2>&1

echo "queue=$QUEUE_NAME git_sha=$(git rev-parse HEAD)"
echo "dataset=$DATASET batch=$B grad_accum=$GA tokens_per_update=$((B * GA * 512))"
START_FREE_GB="$(free_gb)"
echo "disk_free=${START_FREE_GB}GB start_floor=${MIN_START_FREE_GB}GB runtime_floor=${MIN_RUNTIME_FREE_GB}GB"
(( START_FREE_GB >= MIN_START_FREE_GB )) || fail "insufficient start disk"

ACTIVE_TRAINING="$(pgrep -af '[p]ython.*train.py' 2>/dev/null || true)"
[[ -z "$ACTIVE_TRAINING" ]] || fail "another train.py is active: $ACTIVE_TRAINING"

mapfile -t GPU_MEMORY_USED < <(
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{ print int($1) }'
)
(( ${#GPU_MEMORY_USED[@]} == 4 )) || fail "expected four GPUs"
for gpu in "${GPUS[@]}"; do
  (( GPU_MEMORY_USED[$gpu] <= 256 )) || fail "GPU $gpu is busy"
done

for index in "${!NAMES[@]}"; do
  [[ -f "${SCHEDULES[$index]}" ]] || fail "missing schedule: ${SCHEDULES[$index]}"
  [[ ! -e "out-babylm/${NAMES[$index]}" ]] || fail "output exists: ${NAMES[$index]}"
  [[ ! -e "$LOG_DIR/${NAMES[$index]}.train.log" ]] || fail "log exists: ${NAMES[$index]}"
done

"$PY" - "${SCHEDULES[0]}" "${SCHEDULES[2]}" <<'PYCODE'
import hashlib
import json
import pathlib
import sys

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

root = pathlib.Path("data/babylm_100m_officialdev")
actual = {
    "train_bin_sha256": sha256(root / "train.bin"),
    "val_bin_sha256": sha256(root / "val.bin"),
    "tokenizer_sha256": sha256(root / "tokenizer/bpe-16000.json"),
    "word_map_sha256": sha256(root / "train.word_starts.uint8"),
}
for path, seed in zip(sys.argv[1:], (1337, 1338)):
    payload = json.load(open(path, encoding="utf-8"))
    params = payload["parameters"]
    labels = [label for checkpoint in payload["checkpoints"] for label in checkpoint["labels"]]
    assert params["max_iters"] == 4797
    assert params["batch_size"] == 16
    assert params["global_grad_accum"] == 32
    assert params["tokens_per_iter"] == 262144
    assert params["sampler_seed"] == seed
    assert len(payload["checkpoints"]) == 57
    assert sum(label["series"] == "words" for label in labels) == 28
    assert sum(label["series"] == "tokens" for label in labels) == 31
    assert payload["fingerprints"] == actual
print("validated schedules and all data fingerprints")
PYCODE

for index in "${!NAMES[@]}"; do
  launch_run "$index"
  if (( index + 1 < ${#NAMES[@]} )); then
    for ((second = 0; second < STAGGER_SECONDS; second++)); do
      sleep 1
      monitor_once
    done
  fi
done

while any_active; do
  sleep "$POLL_SECONDS"
  monitor_once
done

printf '%s sha=%s log=%s\n' "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$QUEUE_LOG" \
  > "$DONE_MARKER"
echo "==== 100M L32 wave 1 complete ===="
