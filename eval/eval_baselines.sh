#!/usr/bin/env bash
# Evaluate the official BabyLM-2026 GPT-2 baselines on the full strict-track suite.
#   strict       = 100M-word track baseline  (GPT-2, ~98M params)
#   strict-small = 10M-word  track baseline  (GPT-2, ~98M params)
# Both are native HuggingFace GPT2LMHeadModel checkpoints, so there is NO
# nanoGPT->HF conversion step -- we run the official pipeline scripts directly
# on the model directories.
#
# Phases (sequential, one shared GPU slice):
#   1. zero-shot  strict, strict-small   (fast, low-memory)
#   2. GLUE       strict, strict-small   (slow: boolq/multirc/rte/wsc/mrpc/qqp/mnli)
#
# Results land under $EVAL_REPO/results/{strict,strict-small}/ .
#
# Override any path via env: DATA NANO_REPO EVAL_REPO PY MODELS
set -euo pipefail

: "${DATA:=/media/volume/yupei-data}"
: "${NANO_REPO:=$DATA/repo/nanoGPT}"
: "${EVAL_REPO:=$DATA/repo/babylm-eval/strict}"
: "${PY:=$DATA/envs/babylm-eval/bin/python}"
: "${MODELS:=$DATA/models/babylm-baselines}"

export PATH="$(dirname "$PY"):$PATH"                       # pipeline scripts call a bare `python`
cd "$EVAL_REPO"
[ -f "$EVAL_REPO/../.env" ] || : > "$EVAL_REPO/../.env"    # eval_finetuning.sh sources it

ts () { date '+%Y-%m-%d %H:%M:%S'; }

echo "########## PHASE 1: ZERO-SHOT  [$(ts)] ##########"
for m in strict strict-small; do
  echo ">>>>> [$(ts)] ZERO-SHOT START: $m"
  bash scripts/eval_zero_shot.sh "$MODELS/$m" causal
  for task in global_piqa_parallel global_piqa_nonparallel; do
    "$PY" -m evaluation_pipeline.sentence_zero_shot.run \
      --model_path_or_name "$MODELS/$m" --backend causal --task "$task" \
      --data_path "evaluation_data/full_eval/${task}" --save_predictions \
      --revision_name main
  done
  echo ">>>>> [$(ts)] ZERO-SHOT DONE:  $m"
done

echo "########## PHASE 2: GLUE  [$(ts)] ##########"
for m in strict strict-small; do
  echo ">>>>> [$(ts)] GLUE START: $m"
  bash scripts/eval_finetuning.sh --model_path "$MODELS/$m"
  echo ">>>>> [$(ts)] GLUE DONE:  $m"
done

echo "########## SYNC SCOREBOARDS  [$(ts)] ##########"
"$PY" "$NANO_REPO/eval/sync_eval_results.py" strict --csv-model Baseline-Strict \
  --full --glue --backend causal --eval-repo "$EVAL_REPO"
"$PY" "$NANO_REPO/eval/sync_eval_results.py" strict-small --csv-model Baseline-Strict-Small \
  --full --glue --backend causal --eval-repo "$EVAL_REPO"

echo "########## ALL_BASELINES_DONE  [$(ts)] ##########"
