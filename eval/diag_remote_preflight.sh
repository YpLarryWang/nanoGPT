#!/usr/bin/env bash
# Read-only checkpoint/SHA audit before launching diagnosis jobs.

set -euo pipefail

EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA to the deployed diagnosis commit}"
: "${JETSTREAM_HOST:=jetstream-fv-agop}"
: "${VAST_HOST:=vast}"
: "${JETSTREAM_REPO:=/media/volume/yupei-data/repo/nanoGPT}"
: "${VAST_REPO:=/workspace/repo/nanoGPT}"
: "${JETSTREAM_RESULTS:=/media/volume/yupei-data/repo/babylm-eval/strict/results}"
: "${VAST_RESULTS:=/workspace/repo/babylm-eval/strict/results}"
: "${T9_ROOT:=/Volumes/T9/babylm-2026/checkpoints/offdev}"

BASE="bl10m-d512L32-do0.1-gate-offdev-aoaw19-aoat20-u37-b8ga64"
ATTN="bl10m-d512L32-do0.1-gate-attnres8-offdev-aoaw19-aoat20-u37-b8ga64"
JETSTREAM_RUNS=("$BASE" "$ATTN")
VAST_RUNS=(
  "${BASE}-s1338" "${ATTN}-s1338"
  "${BASE}-s1339" "${ATTN}-s1339"
)

check_local_run() {
  local run_dir="$1"
  python3 "$(dirname "$0")/diag_dev_series.py" --run-dir "$run_dir" --plan-only >/dev/null
  echo "local checkpoint plan OK: $run_dir"
}

check_remote_run() {
  local host="$1" repo="$2" results="$3" run="$4" command task leaf root
  command="cd '$repo' && test \"\$(git rev-parse HEAD)\" = '$EXPECTED_SHA' && python3 eval/diag_dev_series.py --run-dir 'out-babylm/$run' --plan-only >/dev/null"
  for task in blimp entity_tracking comps; do
    leaf="$task"
    if [[ "$task" == blimp ]]; then leaf=blimp_filtered; fi
    root="$results/$run/main/zero_shot/causal/$task/$leaf"
    command+=" && test -s '$root/best_temperature_report.txt' && test -s '$root/predictions.json'"
  done
  ssh "$host" "$command"
  echo "remote checkpoint plan + final artifacts OK: $host:$repo/out-babylm/$run"
}

for run in "${JETSTREAM_RUNS[@]}"; do
  check_local_run "$T9_ROOT/jetstream/$run"
  check_remote_run "$JETSTREAM_HOST" "$JETSTREAM_REPO" "$JETSTREAM_RESULTS" "$run"
done
for run in "${VAST_RUNS[@]}"; do
  check_local_run "$T9_ROOT/vast/$run"
  check_remote_run "$VAST_HOST" "$VAST_REPO" "$VAST_RESULTS" "$run"
done

echo "preflight complete: six ladders, six-point plans, remote SHA=$EXPECTED_SHA"
