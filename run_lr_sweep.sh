#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LR sweep on TinyStories — 4 learning rates at ~4 epochs (7200 iters).
#
# All runs log to the SAME wandb project ('tinystories') -> one dashboard you
# can overlay/sort, and each appends one line to results/experiments.jsonl.
#
# No checkpoint series during the sweep (save_iters=[]) to save disk — you're
# comparing FINAL val_loss here, not training dynamics. Once you pick the
# winning LR, do ONE full run at that LR WITH the dense save_iters schedule
# for your downstream analysis.
#
# Usage (on the A100 box, with the conda env active):
#     conda activate nanogpt          # or: source .../activate nanogpt
#     bash run_lr_sweep.sh
# ---------------------------------------------------------------------------
set -u                       # error on undefined vars (NOT -e: keep going if one run fails)
cd "$(dirname "$0")"         # run from the repo root regardless of caller's cwd

# "learning_rate min_lr" pairs  (min_lr = learning_rate / 10)
for pair in "3e-4 3e-5" "6e-4 6e-5" "1e-3 1e-4" "2e-3 2e-4"; do
  set -- $pair; lr=$1; min=$2
  echo "==============================================================="
  echo ">>> sweep run: learning_rate=$lr  min_lr=$min  (max_iters=7200, ~4 epochs)"
  echo "==============================================================="
  python train.py config/train_tinystories.py \
    --learning_rate="$lr" --min_lr="$min" \
    --max_iters=7200 --lr_decay_iters=7200 \
    --save_iters="[]" \
    --out_dir="out-ts-lr$lr" --wandb_run_name="ts-50m-lr$lr-4ep"
done

echo "==============================================================="
echo "sweep complete."
echo "compare:  results/experiments.jsonl   and   wandb project 'tinystories'"
echo "==============================================================="
