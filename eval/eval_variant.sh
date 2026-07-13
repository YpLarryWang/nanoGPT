#!/usr/bin/env bash
# Convert one nanoGPT ablation checkpoint to a HuggingFace model and run the
# BabyLM-2026 *strict* evaluation on it, reusing the official pipeline scripts.
#
# Usage (run from anywhere):
#   bash eval/eval_variant.sh bl100m-ln-mlp-learned          # full zero-shot
#   bash eval/eval_variant.sh bl10m-rms-swiglu-rope --fast   # fast zero-shot (checkpoints)
#   bash eval/eval_variant.sh bl100m-ln-mlp-learned --glue   # + GLUE fine-tuning (slow!)
#
# <variant> is a directory name under out-babylm/ (expects out-babylm/<variant>/ckpt.pt).
# Paths default to the jetstream layout; override any via environment variables:
#   DATA NANO_REPO EVAL_REPO PY HF_ROOT
set -euo pipefail

VARIANT="${1:?usage: eval_variant.sh <variant> [--fast] [--glue]}"; shift || true
FAST=0; GLUE=0
for a in "$@"; do
  case "$a" in
    --fast) FAST=1 ;;
    --glue) GLUE=1 ;;
    *) echo "unknown arg: $a" >&2; exit 1 ;;
  esac
done

: "${DATA:=/media/volume/yupei-data}"
: "${NANO_REPO:=$DATA/repo/nanoGPT}"
: "${EVAL_REPO:=$DATA/repo/babylm-eval/strict}"
: "${PY:=$DATA/envs/babylm-eval/bin/python}"
: "${HF_ROOT:=$DATA/hf-models}"

# the pipeline's scripts invoke a bare `python`; point it at the eval venv
export PATH="$(dirname "$PY"):$PATH"

CKPT="$NANO_REPO/out-babylm/$VARIANT/ckpt.pt"
[ -f "$CKPT" ] || { echo "no checkpoint: $CKPT" >&2; exit 1; }

# 100M models use the 100M tokenizer; everything else uses the 10M tokenizer.
# Match *bl100m-* (substring) so optimizer-prefixed names like muon-bl100m-* are
# caught too; "bl100m" is not a substring of the 10M "bl10m", so 10M names are safe.
case "$VARIANT" in
  *bl100m-*) TOK="$NANO_REPO/data/babylm_100m/tokenizer/bpe-16000.json" ;;
  *)         TOK="$NANO_REPO/data/babylm/tokenizer/bpe-16000.json" ;;
esac
HFDIR="$HF_ROOT/$VARIANT"

echo ">> convert $VARIANT -> $HFDIR"
"$PY" "$NANO_REPO/eval/convert_nanogpt_to_hf.py" --ckpt "$CKPT" --tokenizer "$TOK" --out "$HFDIR"

cd "$EVAL_REPO"
if [ "$FAST" = 1 ]; then
  echo ">> fast zero-shot"
  bash scripts/eval_zero_shot_fast.sh "$HFDIR" main causal
  GLOBAL_PIQA_DATA="evaluation_data/fast_eval"
else
  echo ">> full zero-shot"
  bash scripts/eval_zero_shot.sh "$HFDIR" causal
  GLOBAL_PIQA_DATA="evaluation_data/full_eval"
fi

# GlobalPIQA was added after the original 2026 pipeline release. It is kept in
# a dedicated upstream driver for Hub revisions; local one-checkpoint runs need
# these two explicit calls instead of the all-revisions driver.
echo ">> GlobalPIQA (${GLOBAL_PIQA_DATA})"
for task in global_piqa_parallel global_piqa_nonparallel; do
  "$PY" -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name "$HFDIR" --backend causal --task "$task" \
    --data_path "${GLOBAL_PIQA_DATA}/${task}" --save_predictions \
    --revision_name main
done

if [ "$GLUE" = 1 ]; then
  echo ">> GLUE fine-tuning (slow: boolq/multirc/rte/wsc/mrpc/qqp/mnli)"
  [ -f "$EVAL_REPO/../.env" ] || : > "$EVAL_REPO/../.env"   # eval_finetuning.sh sources it
  bash scripts/eval_finetuning.sh --model_path "$HFDIR"
fi

echo ">> done: $VARIANT  (results under $EVAL_REPO/results/$VARIANT)"
SYNC_ARGS=("$VARIANT" --eval-repo "$EVAL_REPO" --backend causal)
if [ "$FAST" = 1 ]; then SYNC_ARGS+=(--fast); else SYNC_ARGS+=(--full); fi
if [ "$GLUE" = 1 ]; then SYNC_ARGS+=(--glue); fi
echo ">> sync scoreboards"
"$PY" "$NANO_REPO/eval/sync_eval_results.py" "${SYNC_ARGS[@]}"
