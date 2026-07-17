#!/usr/bin/env bash
# Smoke on GPU 0, then run the seed-1338/1339 L32 baseline/AttnRes8 matrix.
# Each formal run uses one GPU; this script does not use DDP.
# Intended launch on the 4x3090 Vast host:
#   tmux new-session -d -s offdev10m-l32-s1338-s1339 \
#     'cd /workspace/nanoGPT && bash run_babylm_offdev_10m_l32_s1338_s1339_4gpu.sh'

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PY="${PY:-/workspace/envs/nanogpt/bin/python}"
DATASET=babylm_officialdev
B=8
GA=64
MAX_ITERS=471
STAGGER_SECONDS="${STAGGER_SECONDS:-45}"
POLL_SECONDS=10
MIN_START_FREE_GB=180
MIN_RUNTIME_FREE_GB=80
QUEUE_NAME=offdev10m-l32-s1338-s1339
LOG_DIR=logs/offdev-10m-l32-s1338-s1339
QUEUE_LOG="$LOG_DIR/${QUEUE_NAME}.queue.log"
DONE_MARKER=results/${QUEUE_NAME}.queue.done
FAILED_MARKER=results/${QUEUE_NAME}.queue.failed
LOCK_DIR=/tmp/${QUEUE_NAME}.lock

GPUS=(0 1 2 3)
SEEDS=(1338 1338 1339 1339)
ATTN_RES_BLOCKS=(0 8 0 8)
SCHEDULES=(
  config/checkpoint_schedules/bl10m-offdev-b8ga64-s1338-dual.json
  config/checkpoint_schedules/bl10m-offdev-b8ga64-s1338-dual.json
  config/checkpoint_schedules/bl10m-offdev-b8ga64-s1339-dual.json
  config/checkpoint_schedules/bl10m-offdev-b8ga64-s1339-dual.json
)
NAMES=(
  bl10m-d512L32-do0.1-gate-offdev-aoaw19-aoat20-u37-b8ga64-s1338
  bl10m-d512L32-do0.1-gate-attnres8-offdev-aoaw19-aoat20-u37-b8ga64-s1338
  bl10m-d512L32-do0.1-gate-offdev-aoaw19-aoat20-u37-b8ga64-s1339
  bl10m-d512L32-do0.1-gate-attnres8-offdev-aoaw19-aoat20-u37-b8ga64-s1339
)

PIDS=("" "" "" "")
ACTIVE=(0 0 0 0)
SMOKE_ROOT=""
FATAL_PATTERN='CUDA out of memory|OutOfMemoryError|checkpoint schedule .*does not match|Traceback|(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)'

fail() {
  echo "FATAL: $*" >&2
  return 1
}

free_gb() {
  df -Pk "$REPO_ROOT" | awk 'NR == 2 { print int($4 / 1024 / 1024) }'
}

check_log_for_fatal_values() {
  local log_path="$1"
  [[ -f "$log_path" ]] || return 0
  if grep -Eiq "$FATAL_PATTERN" "$log_path"; then
    echo "fatal marker found in $log_path" >&2
    grep -Ein "$FATAL_PATTERN" "$log_path" | tail -20 >&2
    return 1
  fi
}

stop_active_runs() {
  local index pid
  for index in "${!PIDS[@]}"; do
    pid="${PIDS[$index]}"
    if [[ "${ACTIVE[$index]}" == 1 && -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "stopping gpu=${GPUS[$index]} pid=$pid name=${NAMES[$index]}" >&2
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
  fi
  if [[ -n "$SMOKE_ROOT" && "$SMOKE_ROOT" == /tmp/${QUEUE_NAME}-smoke.* ]]; then
    rm -rf -- "$SMOKE_ROOT"
  fi
  rmdir "$LOCK_DIR" 2>/dev/null || true
  if (( status != 0 )); then
    printf '%s status=%s log=%s\n' "$(date --iso-8601=seconds)" "$status" "$QUEUE_LOG" \
      > "$FAILED_MARKER"
  fi
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

run_smoke() {
  local name="${NAMES[0]}"
  local smoke_out="$SMOKE_ROOT/$name"
  local smoke_log="$SMOKE_ROOT/$name.log"
  arch_args 0

  echo "[$(date --iso-8601=seconds)] START smoke:$name gpu=0"
  set +e
  CUDA_VISIBLE_DEVICES=0 "$PY" train.py config/train_babylm.py \
    --dataset="$DATASET" \
    --max_iters=2 --lr_decay_iters=2 --warmup_iters=1 \
    --eval_interval=1 --eval_iters=2 --eval_batch_size=32 --log_interval=1 \
    --use_rmsnorm=True --use_swiglu=True --swiglu_mult=2.6666666666666665 \
    --use_rope=True --use_attn_gate=True \
    --use_muon=False --use_hybrid=False \
    --sampler=shuffle --sampler_seed=1338 --dropout=0.1 \
    "${ARCH_ARGS[@]}" \
    --batch_size="$B" --gradient_accumulation_steps="$GA" \
    --seed=1338 \
    --wandb_log=False --wandb_run_name="${name}-smoke" \
    --experiment_log_path="$SMOKE_ROOT/experiments.jsonl" \
    --out_dir="$smoke_out" 2>&1 | tee "$smoke_log"
  local status=${PIPESTATUS[0]}
  set -e
  (( status == 0 )) || fail "smoke exited with status $status"
  check_log_for_fatal_values "$smoke_log" || fail "smoke emitted a fatal marker"
  echo "[$(date --iso-8601=seconds)] COMPLETE smoke:$name"
}

launch_formal() {
  local index="$1"
  local gpu="${GPUS[$index]}"
  local seed="${SEEDS[$index]}"
  local name="${NAMES[$index]}"
  local schedule="${SCHEDULES[$index]}"
  local attn_res_block="${ATTN_RES_BLOCKS[$index]}"
  local out_dir="out-babylm/$name"
  local run_log="$LOG_DIR/$name.train.log"
  arch_args "$attn_res_block"

  echo "[$(date --iso-8601=seconds)] START formal:$name gpu=$gpu seed=$seed log=$run_log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py config/train_babylm.py \
    --dataset="$DATASET" \
    --max_iters="$MAX_ITERS" --lr_decay_iters="$MAX_ITERS" --warmup_iters=40 \
    --eval_interval=50 --eval_iters=50 --eval_batch_size=32 \
    --checkpoint_schedule="$schedule" \
    --use_rmsnorm=True --use_swiglu=True --swiglu_mult=2.6666666666666665 \
    --use_rope=True --use_attn_gate=True \
    --use_muon=False --use_hybrid=False \
    --sampler=shuffle --sampler_seed="$seed" --dropout=0.1 \
    "${ARCH_ARGS[@]}" \
    --batch_size="$B" --gradient_accumulation_steps="$GA" \
    --seed="$seed" \
    --wandb_log=True --wandb_project=babylm \
    --wandb_run_name="$name" --out_dir="$out_dir" > "$run_log" 2>&1 &
  PIDS[$index]=$!
  ACTIVE[$index]=1
  echo "launched gpu=$gpu pid=${PIDS[$index]} name=$name"
}

monitor_once() {
  local available index pid status run_log
  available="$(free_gb)"
  (( available >= MIN_RUNTIME_FREE_GB )) || fail \
    "disk fell below runtime floor: ${available}GB < ${MIN_RUNTIME_FREE_GB}GB"

  for index in "${!PIDS[@]}"; do
    [[ "${ACTIVE[$index]}" == 1 ]] || continue
    pid="${PIDS[$index]}"
    run_log="$LOG_DIR/${NAMES[$index]}.train.log"
    check_log_for_fatal_values "$run_log" || fail \
      "formal run emitted a fatal marker: ${NAMES[$index]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      set +e
      wait "$pid"
      status=$?
      set -e
      (( status == 0 )) || fail \
        "formal run exited with status $status: ${NAMES[$index]}"
      check_log_for_fatal_values "$run_log" || fail \
        "formal run completed with a fatal marker: ${NAMES[$index]}"
      ACTIVE[$index]=0
      echo "[$(date --iso-8601=seconds)] COMPLETE formal:${NAMES[$index]} gpu=${GPUS[$index]}"
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

[[ -n "${TMUX:-}" ]] || fail "launch this queue inside tmux"
[[ -x "$PY" ]] || fail "training Python is not executable: $PY"
[[ "$STAGGER_SECONDS" =~ ^[0-9]+$ ]] || fail "STAGGER_SECONDS must be a nonnegative integer"
[[ ! -e "$QUEUE_LOG" ]] || fail "queue log already exists: $QUEUE_LOG"
[[ ! -e "$DONE_MARKER" ]] || fail "queue already completed: $DONE_MARKER"
[[ ! -e "$FAILED_MARKER" ]] || fail "previous queue failure needs review: $FAILED_MARKER"
mkdir "$LOCK_DIR" 2>/dev/null || fail "another $QUEUE_NAME queue appears to be active"

mkdir -p "$LOG_DIR" results
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec > >(tee -a "$QUEUE_LOG") 2>&1
SMOKE_ROOT="$(mktemp -d /tmp/${QUEUE_NAME}-smoke.XXXXXX)"

echo "queue=$QUEUE_NAME"
echo "git_sha=$(git rev-parse HEAD)"
echo "python=$PY"
echo "dataset=$DATASET batch=$B grad_accum=$GA tokens_per_update=$((B * GA * 512))"
echo "eval_batch_size=32 stagger_seconds=$STAGGER_SECONDS smoke_root=$SMOKE_ROOT"

START_FREE_GB="$(free_gb)"
echo "disk_free=${START_FREE_GB}GB required_at_start=${MIN_START_FREE_GB}GB runtime_floor=${MIN_RUNTIME_FREE_GB}GB"
(( START_FREE_GB >= MIN_START_FREE_GB )) || fail \
  "only ${START_FREE_GB}GB free; refusing to start below ${MIN_START_FREE_GB}GB"

ACTIVE_TRAINING="$(pgrep -af '[p]ython.*train.py' 2>/dev/null || true)"
if [[ -n "$ACTIVE_TRAINING" ]]; then
  echo "$ACTIVE_TRAINING" >&2
  fail "another train.py process is already running"
fi

mapfile -t GPU_MEMORY_USED < <(
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{ print int($1) }'
)
(( ${#GPU_MEMORY_USED[@]} == 4 )) || fail "expected exactly four visible GPUs"
for gpu in "${GPUS[@]}"; do
  echo "gpu=$gpu memory_used=${GPU_MEMORY_USED[$gpu]}MiB"
  (( GPU_MEMORY_USED[$gpu] <= 256 )) || fail \
    "GPU $gpu is not idle (${GPU_MEMORY_USED[$gpu]}MiB in use)"
done

CUDA_VISIBLE_DEVICES=0 "$PY" -c '
import torch
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported(), "GPU does not support BF16"
print(f"torch={torch.__version__} cuda={torch.version.cuda} bf16=True")
'

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  [[ -f "$HOME/.netrc" ]] && grep -q 'machine api.wandb.ai' "$HOME/.netrc" || \
    fail "W&B authentication is missing"
fi

for index in "${!NAMES[@]}"; do
  name="${NAMES[$index]}"
  [[ -f "${SCHEDULES[$index]}" ]] || fail "missing schedule: ${SCHEDULES[$index]}"
  [[ ! -e "out-babylm/$name" ]] || fail "output already exists: out-babylm/$name"
  [[ ! -e "$LOG_DIR/$name.train.log" ]] || fail "run log already exists: $LOG_DIR/$name.train.log"
  if grep -q "\"run_name\": \"${name}\"" results/experiments.jsonl 2>/dev/null; then
    fail "run already recorded in results/experiments.jsonl: $name"
  fi
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

root = pathlib.Path("data/babylm_officialdev")
actual = {
    "train_bin_sha256": sha256(root / "train.bin"),
    "val_bin_sha256": sha256(root / "val.bin"),
    "tokenizer_sha256": sha256(root / "tokenizer/bpe-16000.json"),
    "word_map_sha256": sha256(root / "train.word_starts.uint8"),
}
for path, seed in zip(sys.argv[1:], (1338, 1339)):
    payload = json.load(open(path, encoding="utf-8"))
    params = payload["parameters"]
    labels = [label for checkpoint in payload["checkpoints"] for label in checkpoint["labels"]]
    assert params["max_iters"] == 471
    assert params["block_size"] == 512
    assert params["batch_size"] == 8
    assert params["global_grad_accum"] == 64
    assert params["world_size"] == 1
    assert params["tokens_per_iter"] == 262144
    assert params["sampler_seed"] == seed
    assert sum(label["series"] == "words" for label in labels) == 19
    assert sum(label["series"] == "tokens" for label in labels) == 20
    assert len(payload["checkpoints"]) == 37
    assert payload["fingerprints"] == actual
print("validated s1338/s1339 schedules and all data fingerprints")
PYCODE

echo "==== phase 1: GPU-0 L32 baseline 2-update smoke ===="
run_smoke

echo "==== phase 2: four independent L32 formal runs ===="
for index in "${!NAMES[@]}"; do
  launch_formal "$index"
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
echo "==== seed-1338/1339 offdev 10M L32 4-GPU queue complete ===="
