#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# (b) Multi-GPU sweep launcher: ONE run per GPU, in parallel; queue the rest.
#
# Auto-detects GPUs (e.g. your 2x 3090 -> 2-wide, or 1x A6000 -> sequential), or
# pin specific cards with the GPUS env var, e.g.  GPUS=0,1 ./run_sweep_multigpu.sh
# Each run is pinned with CUDA_VISIBLE_DEVICES, gets its own out_dir, its own
# wandb_run_name, and its own log file in logs/. Runs survive disconnect if you
# launch this inside tmux.
#
# Usage:
#     ./run_sweep_multigpu.sh            # runs the SWEEP defined below
# Monitor:
#     tail -f logs/<label>.log           # one run
#     nvidia-smi -l 5                    # all GPUs
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate 2>/dev/null || true

# ---- EDIT YOUR SWEEP HERE: "label|extra train.py overrides" -----------------
RUNS=(
  "lr3e-4|--learning_rate=3e-4 --min_lr=3e-5"
  "lr6e-4|--learning_rate=6e-4 --min_lr=6e-5"
  "lr1e-3|--learning_rate=1e-3 --min_lr=1e-4"
  "lr2e-3|--learning_rate=2e-3 --min_lr=2e-4"
)
# args shared by every run (config file first):
COMMON_ARGS="config/train_tinystories.py --max_iters=7200 --lr_decay_iters=7200 --wandb_log=True"
TAG="$(date +%m%d)"   # appended to wandb run names so reruns don't collide
# -----------------------------------------------------------------------------

mkdir -p logs
# GPUs to use: comma-list via GPUS env (e.g. GPUS=0,1), else auto-detect all visible.
if [ -n "${GPUS:-}" ]; then
  IFS=',' read -r -a GPU_LIST <<< "$GPUS"
else
  mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
fi
[ "${#GPU_LIST[@]}" -eq 0 ] && GPU_LIST=(0)   # fallback if nvidia-smi unavailable
NGPU=${#GPU_LIST[@]}
echo "Detected $NGPU GPU(s): ${GPU_LIST[*]}   |   ${#RUNS[@]} run(s) queued"

declare -A GPU_PID    # gpu index -> pid of the run currently on it (unset = free)

launch() {  # <gpu> <label> <args>
  local gpu="$1" label="$2" args="$3"
  echo ">>> [GPU $gpu] start $label   (log: logs/$label.log)"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python train.py $COMMON_ARGS $args \
      --out_dir="out-$label" --wandb_run_name="ts-50m-$label-$TAG" \
      > "logs/$label.log" 2>&1 &
  GPU_PID["$gpu"]=$!
}

# schedule: launch each run on the first free GPU; wait when all GPUs are busy
i=0
while [ "$i" -lt "${#RUNS[@]}" ]; do
  free=""
  for g in "${GPU_LIST[@]}"; do
    pid="${GPU_PID[$g]:-}"
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then free="$g"; break; fi
  done
  if [ -z "$free" ]; then sleep 5; continue; fi   # all GPUs busy -> wait
  IFS='|' read -r label args <<< "${RUNS[$i]}"
  launch "$free" "$label" "$args"
  i=$((i + 1))
  sleep 2   # small stagger so two launches don't grab the same GPU
done

echo "all ${#RUNS[@]} run(s) launched across $NGPU GPU(s); waiting for completion..."
wait
echo "=== sweep complete — logs in logs/, compare in wandb ==="
