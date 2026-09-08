#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
PY=/workspace/envs/train/bin/python
"$PY" train.py config/train_babylm.py --dataset=babylm_officialdev \
 --n_layer=32 --n_head=8 --n_embd=512 --use_rmsnorm=True --use_swiglu=True \
 --swiglu_mult=2.6666666666666665 --use_rope=True --use_attn_gate=True \
 --use_attn_res=True --use_static_attn_res=True --attn_res_block_size=8 \
 --sampler=shuffle --sampler_seed=1337 --seed=1337 --dropout=0.1 \
 --batch_size=8 --gradient_accumulation_steps=64 --max_iters=2 --lr_decay_iters=2 --warmup_iters=0 \
 --eval_iters=2 --eval_interval=1 --eval_batch_size=32 --log_interval=1 \
 --endpoint_only=True --wandb_log=False --out_dir=out-babylm/_cr-smoke-static
