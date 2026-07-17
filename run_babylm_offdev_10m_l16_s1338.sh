#!/usr/bin/env bash
# Smoke and then run the seed-1338 10M offdev L16 baseline/AttnRes4 pair.
# Intended launch:
#   tmux new-session -d -s offdev10m-l16-s1338 \
#     'cd /media/volume/yupei-data/repo/nanoGPT && bash run_babylm_offdev_10m_l16_s1338.sh'

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PY="${PY:-/media/volume/yupei-data/envs/nanogpt/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATASET=babylm_officialdev
SEED=1338
B=8
GA=64
MAX_ITERS=471
SCHEDULE=config/checkpoint_schedules/bl10m-offdev-b8ga64-s1338-dual.json
MIN_FREE_GB=220
QUEUE_NAME=offdev10m-l16-s1338
LOG_DIR=logs/offdev-10m-l16-s1338
QUEUE_LOG="$LOG_DIR/${QUEUE_NAME}.queue.log"
DONE_MARKER=results/${QUEUE_NAME}.queue.done
FAILED_MARKER=results/${QUEUE_NAME}.queue.failed
LOCK_DIR=/tmp/${QUEUE_NAME}.lock

NAMES=(
  bl10m-d512L16-do0.1-gate-offdev-aoaw19-aoat20-u37-b8ga64-s1338
  bl10m-d512L16-do0.1-gate-attnres4-offdev-aoaw19-aoat20-u37-b8ga64-s1338
)

fail() {
  echo "FATAL: $*" >&2
  return 1
}

free_gb() {
  df -Pk "$REPO_ROOT" | awk 'NR == 2 { print int($4 / 1024 / 1024) }'
}

require_disk() {
  local available
  available="$(free_gb)"
  echo "disk_free=${available}GB required=${MIN_FREE_GB}GB"
  (( available >= MIN_FREE_GB )) || fail \
    "only ${available}GB free; refusing to continue below ${MIN_FREE_GB}GB"
}

check_log_for_fatal_values() {
  local log_path="$1"
  local pattern='CUDA out of memory|OutOfMemoryError|checkpoint schedule .*does not match|Traceback|(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)'
  if grep -Eiq "$pattern" "$log_path"; then
    echo "fatal marker found in $log_path" >&2
    grep -Ein "$pattern" "$log_path" | tail -20 >&2
    return 1
  fi
}

run_logged() {
  local label="$1"
  local log_path="$2"
  shift 2

  echo "[$(date --iso-8601=seconds)] START $label"
  echo "log=$log_path"
  set +e
  "$@" 2>&1 | tee "$log_path"
  local status=${PIPESTATUS[0]}
  set -e
  (( status == 0 )) || fail "$label exited with status $status"
  check_log_for_fatal_values "$log_path" || fail "$label emitted a fatal marker"
  echo "[$(date --iso-8601=seconds)] COMPLETE $label"
}

arch_args() {
  local attn_res_block="$1"
  ARCH_ARGS=(--n_layer=16 --n_embd=512 --n_head=8)
  if (( attn_res_block == 0 )); then
    ARCH_ARGS+=(--use_attn_res=False)
  else
    ARCH_ARGS+=(--use_attn_res=True --attn_res_block_size="$attn_res_block")
  fi
}

run_smoke() {
  local name="$1"
  local attn_res_block="$2"
  local smoke_out="$SMOKE_ROOT/$name"
  local smoke_log="$SMOKE_ROOT/$name.log"
  arch_args "$attn_res_block"

  run_logged "smoke:$name" "$smoke_log" \
    "$PY" train.py config/train_babylm.py \
      --dataset="$DATASET" \
      --max_iters=2 --lr_decay_iters=2 --warmup_iters=1 \
      --eval_interval=1 --eval_iters=2 --eval_batch_size=32 --log_interval=1 \
      --use_rmsnorm=True --use_swiglu=True --swiglu_mult=2.6666666666666665 \
      --use_rope=True --use_attn_gate=True \
      --use_muon=False --use_hybrid=False \
      --sampler=shuffle --sampler_seed="$SEED" --dropout=0.1 \
      "${ARCH_ARGS[@]}" \
      --batch_size="$B" --gradient_accumulation_steps="$GA" \
      --seed="$SEED" \
      --wandb_log=False --wandb_run_name="${name}-smoke" \
      --experiment_log_path="$SMOKE_ROOT/experiments.jsonl" \
      --out_dir="$smoke_out"
}

run_formal() {
  local name="$1"
  local attn_res_block="$2"
  local out_dir="out-babylm/$name"
  local run_log="$LOG_DIR/$name.train.log"
  arch_args "$attn_res_block"

  require_disk
  run_logged "formal:$name" "$run_log" \
    "$PY" train.py config/train_babylm.py \
      --dataset="$DATASET" \
      --max_iters="$MAX_ITERS" --lr_decay_iters="$MAX_ITERS" --warmup_iters=40 \
      --eval_interval=50 --eval_iters=50 --eval_batch_size=32 \
      --checkpoint_schedule="$SCHEDULE" \
      --use_rmsnorm=True --use_swiglu=True --swiglu_mult=2.6666666666666665 \
      --use_rope=True --use_attn_gate=True \
      --use_muon=False --use_hybrid=False \
      --sampler=shuffle --sampler_seed="$SEED" --dropout=0.1 \
      "${ARCH_ARGS[@]}" \
      --batch_size="$B" --gradient_accumulation_steps="$GA" \
      --seed="$SEED" \
      --wandb_log=True --wandb_project=babylm \
      --wandb_run_name="$name" --out_dir="$out_dir"
}

[[ -n "${TMUX:-}" ]] || fail "launch this queue inside tmux"
[[ -x "$PY" ]] || fail "training Python is not executable: $PY"
[[ -f "$SCHEDULE" ]] || fail "missing checkpoint schedule: $SCHEDULE"
[[ ! -e "$QUEUE_LOG" ]] || fail "queue log already exists: $QUEUE_LOG"
[[ ! -e "$DONE_MARKER" ]] || fail "queue already completed: $DONE_MARKER"
[[ ! -e "$FAILED_MARKER" ]] || fail "previous queue failure needs review: $FAILED_MARKER"
mkdir "$LOCK_DIR" 2>/dev/null || fail "another $QUEUE_NAME queue appears to be active"

SMOKE_ROOT="$(mktemp -d /tmp/${QUEUE_NAME}-smoke.XXXXXX)"
cleanup() {
  local status=$?
  if [[ "$SMOKE_ROOT" == /tmp/${QUEUE_NAME}-smoke.* ]]; then
    rm -rf -- "$SMOKE_ROOT"
  fi
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

echo "queue=$QUEUE_NAME"
echo "git_sha=$(git rev-parse HEAD)"
echo "python=$PY"
echo "dataset=$DATASET seed=$SEED batch=$B grad_accum=$GA tokens_per_update=$((B * GA * 512))"
echo "eval_batch_size=32 schedule=$SCHEDULE smoke_root=$SMOKE_ROOT"

require_disk
ACTIVE_TRAINING="$(pgrep -af '[p]ython.*train.py' 2>/dev/null || true)"
if [[ -n "$ACTIVE_TRAINING" ]]; then
  echo "$ACTIVE_TRAINING" >&2
  fail "another train.py process is already running"
fi

GPU_MEMORY_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR == 1 { print int($1) }')"
echo "gpu_memory_used=${GPU_MEMORY_USED}MiB"
(( GPU_MEMORY_USED <= 512 )) || fail "GPU is not idle (${GPU_MEMORY_USED}MiB in use)"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  [[ -f "$HOME/.netrc" ]] && grep -q 'machine api.wandb.ai' "$HOME/.netrc" || \
    fail "W&B authentication is missing"
fi

for name in "${NAMES[@]}"; do
  [[ ! -e "out-babylm/$name" ]] || fail "output already exists: out-babylm/$name"
  [[ ! -e "$LOG_DIR/$name.train.log" ]] || fail "run log already exists: $LOG_DIR/$name.train.log"
  if grep -q "\"run_name\": \"${name}\"" results/experiments.jsonl 2>/dev/null; then
    fail "run already recorded in results/experiments.jsonl: $name"
  fi
done

"$PY" - "$SCHEDULE" <<'PYCODE'
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

payload = json.load(open(sys.argv[1], encoding="utf-8"))
params = payload["parameters"]
labels = [label for checkpoint in payload["checkpoints"] for label in checkpoint["labels"]]
assert params["max_iters"] == 471
assert params["block_size"] == 512
assert params["batch_size"] == 8
assert params["global_grad_accum"] == 64
assert params["world_size"] == 1
assert params["tokens_per_iter"] == 262144
assert params["sampler_seed"] == 1338
assert sum(label["series"] == "words" for label in labels) == 19
assert sum(label["series"] == "tokens" for label in labels) == 20
assert len(payload["checkpoints"]) == 37

root = pathlib.Path("data/babylm_officialdev")
actual = {
    "train_bin_sha256": sha256(root / "train.bin"),
    "val_bin_sha256": sha256(root / "val.bin"),
    "tokenizer_sha256": sha256(root / "tokenizer/bpe-16000.json"),
    "word_map_sha256": sha256(root / "train.word_starts.uint8"),
}
assert actual == payload["fingerprints"], (actual, payload["fingerprints"])
print("validated s1338 B8/GA64 schedule and all data fingerprints")
PYCODE

echo "==== phase 1: two seed-1338 L16 2-update smoke tests ===="
run_smoke "${NAMES[0]}" 0
run_smoke "${NAMES[1]}" 4

echo "==== phase 2: seed-1338 L16 formal pair ===="
run_formal "${NAMES[0]}" 0
run_formal "${NAMES[1]}" 4

printf '%s sha=%s log=%s\n' "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$QUEUE_LOG" \
  > "$DONE_MARKER"
echo "==== seed-1338 offdev 10M L16 pair complete ===="
