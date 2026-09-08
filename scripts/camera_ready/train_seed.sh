#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SEED=${1:?seed required}; GPU=${2:?GPU required}
PY=${PY:-/workspace/envs/train/bin/python}
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=4
NAME="bl10m-d512L32-do0.1-gate-attnres8-static-offdev-endpoint-b8ga64-s${SEED}"
SCHEDULE=config/checkpoint_schedules/bl10m-offdev-b8ga64-dual.json
if [[ "$SEED" != 1337 ]]; then SCHEDULE="config/checkpoint_schedules/bl10m-offdev-b8ga64-s${SEED}-dual.json"; fi
[[ -s data/babylm_officialdev/camera_ready_train_data.verified ]]
[[ ! -e "out-babylm/$NAME" ]]
mkdir -p logs/camera-ready results/camera-ready
"$PY" train.py config/train_babylm.py \
 --dataset=babylm_officialdev --max_iters=471 --lr_decay_iters=471 --warmup_iters=40 \
 --n_layer=32 --n_embd=512 --n_head=8 --use_rmsnorm=True --use_swiglu=True \
 --swiglu_mult=2.6666666666666665 --use_rope=True --use_attn_gate=True \
 --use_attn_res=True --use_static_attn_res=True --attn_res_block_size=8 \
 --use_muon=False --use_hybrid=False --sampler=shuffle --sampler_seed="$SEED" \
 --dropout=0.1 --batch_size=8 --gradient_accumulation_steps=64 --seed="$SEED" \
 --eval_interval=50 --eval_iters=50 --eval_batch_size=32 \
 --checkpoint_schedule="$SCHEDULE" --endpoint_only=True \
 --wandb_log=True --wandb_project=babylm --wandb_run_name="$NAME" --wandb_run_id="$NAME" --out_dir="out-babylm/$NAME" \
 > "logs/camera-ready/$NAME.train.log" 2>&1
"$PY" - "$NAME" <<'PY'
from pathlib import Path
import json,sys
p=Path('out-babylm')/sys.argv[1];d=json.loads((p/'checkpoint_manifest.json').read_text())
assert set(d['roles'])=={'final'},d['roles']
assert len(list(p.glob('*.pt')))==1
PY
touch "results/camera-ready/$NAME.train.done"
