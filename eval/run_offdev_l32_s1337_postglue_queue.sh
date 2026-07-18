#!/usr/bin/env bash
set -Eeuo pipefail

NANO_ROOT="${NANO_ROOT:-/media/volume/yupei-data/repo/nanoGPT}"
EVAL_ROOT="${EVAL_ROOT:-/media/volume/yupei-data/repo/babylm-eval/strict}"
EVAL_PYTHON="${EVAL_PYTHON:-/media/volume/yupei-data/envs/babylm-eval/bin/python}"
DATA_ROOT="${DATA_ROOT:-${NANO_ROOT}/data/babylm_officialdev}"
TOKENIZER="${TOKENIZER:-${DATA_ROOT}/tokenizer/bpe-16000.json}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
GPU_IDLE_TIMEOUT="${GPU_IDLE_TIMEOUT:-900}"

GLUE_VARIANT="bl10m-d512L32-do0.1-gate-attnres8-offdev-aoaw19-aoat20-u37-b8ga64"
BASE_VARIANT="bl10m-d512L32-do0.1-gate-offdev-aoaw19-aoat20-u37-b8ga64"
ATTN_VARIANT="${GLUE_VARIANT}"

GLUE_SESSION="offdev10m-l32-attnres8-s1337-glue-official"
GLUE_DONE="${NANO_ROOT}/results/${GLUE_VARIANT}.glue-accuracy.done"
GLUE_FAILED="${NANO_ROOT}/results/${GLUE_VARIANT}.glue-accuracy.failed"
GLUE_LOG="${NANO_ROOT}/logs/glue-offdev-accuracy/${GLUE_VARIANT}.log"
GLUE_RESULTS="${EVAL_ROOT}/results/${GLUE_VARIANT}/main/finetune"

BASE_RUN="${NANO_ROOT}/out-babylm/${BASE_VARIANT}"
BASE_CACHE="/media/volume/yupei-data/hf-models/local-aoa-bl10m-offdev-l32-baseline-s1337-words-fp16"
BASE_OUTPUT="/media/volume/yupei-data/aoa-results/2026-07-18-bl10m-offdev-l32-baseline-s1337-words-minctx0"
BASE_RESULT="${BASE_OUTPUT}/${BASE_VARIANT}"
BASE_LOG="${BASE_OUTPUT}/run-recovery.log"

ATTN_RUN="${NANO_ROOT}/out-babylm/${ATTN_VARIANT}"
ATTN_CACHE="/media/volume/yupei-data/hf-models/local-aoa-bl10m-offdev-l32-attnres8-s1337-words-fp16"
ATTN_OUTPUT="/media/volume/yupei-data/aoa-results/2026-07-18-bl10m-offdev-l32-attnres8-s1337-words-minctx0"
ATTN_RESULT="${ATTN_OUTPUT}/${ATTN_VARIANT}"
ATTN_LOG="${ATTN_OUTPUT}/run.log"

QUEUE_ID="offdev-l32-s1337-postglue"
QUEUE_LOG_DIR="${NANO_ROOT}/logs/${QUEUE_ID}"
QUEUE_LOG="${QUEUE_LOG_DIR}/queue.log"
STATUS_FILE="${NANO_ROOT}/results/${QUEUE_ID}.status"
DONE_MARKER="${NANO_ROOT}/results/${QUEUE_ID}.done"
FAILED_MARKER="${NANO_ROOT}/results/${QUEUE_ID}.failed"
CURRENT_STAGE="initializing"

mkdir -p "${QUEUE_LOG_DIR}" "${NANO_ROOT}/results"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

write_status() {
  CURRENT_STAGE="$1"
  printf 'time=%s\nstage=%s\n' "$(date -Is)" "${CURRENT_STAGE}" > "${STATUS_FILE}"
  echo "[$(date -Is)] stage=${CURRENT_STAGE}"
}

on_error() {
  local rc=$?
  trap - ERR
  printf 'time=%s\nstage=%s\nexit_code=%s\n' \
    "$(date -Is)" "${CURRENT_STAGE}" "${rc}" > "${FAILED_MARKER}"
  echo "[$(date -Is)] FAILED stage=${CURRENT_STAGE} exit_code=${rc}" >&2
  exit "${rc}"
}
trap on_error ERR

verify_glue() {
  if grep -Eqi 'Traceback|OutOfMemory|CUDA error|Killed' "${GLUE_LOG}"; then
    echo "GLUE log contains a fatal runtime signal" >&2
    return 1
  fi
  if grep -Eqi '(^|[^[:alnum:]_])nan([^[:alnum:]_]|$)' "${GLUE_LOG}"; then
    echo "GLUE log contains a standalone NaN signal" >&2
    return 1
  fi

  "${EVAL_PYTHON}" - "${GLUE_RESULTS}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
tasks = {"boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli"}
missing = [task for task in sorted(tasks) if not (root / task / "results.txt").is_file()]
if missing:
    raise SystemExit(f"missing GLUE results.txt files: {missing}")
for task in tasks:
    path = root / task / "results.txt"
    if path.stat().st_size == 0:
        raise SystemExit(f"empty GLUE results file: {path}")

summary_path = root / "accuracy_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
accuracies = summary.get("task_accuracy", {})
if set(accuracies) != tasks:
    raise SystemExit(f"GLUE summary tasks mismatch: {sorted(accuracies)}")
for task, value in accuracies.items():
    if not math.isfinite(float(value)):
        raise SystemExit(f"non-finite GLUE accuracy: {task}={value}")
score = float(summary["superglue_ld"])
if not math.isfinite(score):
    raise SystemExit(f"non-finite superglue_ld: {score}")
print(json.dumps(summary, sort_keys=True))
PY
}

verify_aoa() {
  local result_dir="$1"
  local log_file="$2"
  local run_dir="$3"
  test -s "${result_dir}/surprisal.json"
  test -s "${result_dir}/aoa_score.json"
  grep -q 'local AoA complete: checkpoints=19' "${log_file}"
  if grep -Eqi 'Traceback|CUDA error|OutOfMemory|Killed|non-finite|incomplete AoA results' "${log_file}"; then
    echo "AoA log contains an invalid completion signal: ${log_file}" >&2
    return 1
  fi

  "${EVAL_PYTHON}" - "${NANO_ROOT}" "${run_dir}" "${result_dir}" <<'PY'
import json
import math
import sys
from pathlib import Path

nano_root = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
result_dir = Path(sys.argv[3])
sys.path.insert(0, str(nano_root))

from eval.run_local_aoa import local_step_plan, validate_complete_results

manifest = json.loads(
    (run_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
)
plan = local_step_plan(run_dir, manifest, "words")
if len(plan) != 19:
    raise SystemExit(f"expected 19 word checkpoints, found {len(plan)}")

results = json.loads(
    (result_dir / "surprisal.json").read_text(encoding="utf-8")
)
validate_complete_results(plan, results, expected_per_step=8005)

score = json.loads(
    (result_dir / "aoa_score.json").read_text(encoding="utf-8")
)
curve_fitness = float(score["aoa"]["curve_fitness"])
if not math.isfinite(curve_fitness):
    raise SystemExit(f"non-finite curve_fitness: {curve_fitness}")
print(
    f"validated AoA: checkpoints={len(plan)} "
    f"rows={len(results['results'])} curve_fitness={curve_fitness}"
)
PY
}

wait_for_gpu_idle() {
  local waited=0
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader \
    | grep -q '[0-9]'; do
    if (( waited >= GPU_IDLE_TIMEOUT )); then
      echo "GPU still has a compute process after ${waited}s" >&2
      return 1
    fi
    sleep 10
    waited=$((waited + 10))
  done
}

run_aoa() {
  local run_dir="$1"
  local cache_dir="$2"
  local output_dir="$3"
  local log_file="$4"
  shift 4
  mkdir -p "${output_dir}"
  "$@" "${EVAL_PYTHON}" "${NANO_ROOT}/eval/run_local_aoa.py" \
    --run-dir "${run_dir}" \
    --eval-root "${EVAL_ROOT}" \
    --tokenizer "${TOKENIZER}" \
    --cache-dir "${cache_dir}" \
    --output-dir "${output_dir}" \
    --series words \
    --dtype float16 \
    --python "${EVAL_PYTHON}" \
    --min-context 0 \
    --resume > "${log_file}" 2>&1
}

test ! -e "${DONE_MARKER}"
test ! -e "${FAILED_MARKER}"
test -x "${EVAL_PYTHON}"
test -f "${TOKENIZER}"
test -f "${BASE_RUN}/checkpoint_manifest.json"
test -f "${ATTN_RUN}/checkpoint_manifest.json"

write_status waiting_for_attnres_glue_s1337
while test ! -f "${GLUE_DONE}"; do
  if test -f "${GLUE_FAILED}"; then
    echo "current GLUE runner produced a failed marker" >&2
    false
  fi
  if ! tmux has-session -t "${GLUE_SESSION}" 2>/dev/null; then
    if test -f "${GLUE_DONE}"; then
      break
    else
      echo "current GLUE tmux disappeared without a done marker" >&2
      false
    fi
  fi
  sleep "${WAIT_SECONDS}"
done

write_status validating_attnres_glue_s1337
verify_glue

write_status preparing_baseline_aoa_recovery_s1337
if test -f "${BASE_OUTPUT}/run.log"; then
  attempt_dir="${BASE_OUTPUT}/failed-attempt-20260718-concurrent-glue-cuda"
  mkdir -p "${attempt_dir}"
  for stale in "${BASE_RESULT}/surprisal.json" "${BASE_RESULT}/aoa_score.json" "${BASE_OUTPUT}/run.log"; do
    if test -e "${stale}"; then
      mv "${stale}" "${attempt_dir}/"
    fi
  done
  if test -f "${BASE_RESULT}/resume/surprisal.json"; then
    cp -p "${BASE_RESULT}/resume/surprisal.json" "${attempt_dir}/resume-surprisal.json"
  fi
fi

wait_for_gpu_idle

write_status running_baseline_aoa_recovery_s1337
run_aoa "${BASE_RUN}" "${BASE_CACHE}" "${BASE_OUTPUT}" "${BASE_LOG}" \
  env CUDA_LAUNCH_BLOCKING=1
verify_aoa "${BASE_RESULT}" "${BASE_LOG}" "${BASE_RUN}"

wait_for_gpu_idle

write_status running_attnres8_aoa_s1337
run_aoa "${ATTN_RUN}" "${ATTN_CACHE}" "${ATTN_OUTPUT}" "${ATTN_LOG}" env
verify_aoa "${ATTN_RESULT}" "${ATTN_LOG}" "${ATTN_RUN}"

write_status complete
printf 'time=%s\nstage=complete\nbaseline_aoa=%s\nattnres_aoa=%s\n' \
  "$(date -Is)" "${BASE_RESULT}/aoa_score.json" "${ATTN_RESULT}/aoa_score.json" \
  > "${DONE_MARKER}"
echo "[$(date -Is)] queue complete"
