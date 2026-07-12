#!/usr/bin/env bash
# Export one hybrid checkpoint with full bidirectional attention, evaluate its
# shifted masked-next-token objective with the official MNTP backend, then run
# the standard GLUE suite on the same bidirectional backbone.
#
# Usage: bash eval/eval_bidirectional_variant.sh <causal-variant>

set -euo pipefail

SOURCE="${1:?usage: eval_bidirectional_variant.sh <causal-variant>}"
VARIANT="${SOURCE}-bidir"

: "${DATA:=/media/volume/yupei-data}"
: "${NANO_REPO:=$DATA/repo/nanoGPT}"
: "${EVAL_REPO:=$DATA/repo/babylm-eval/strict}"
: "${PY:=$DATA/envs/babylm-eval/bin/python}"
: "${HF_ROOT:=$DATA/hf-models}"

CKPT="$NANO_REPO/out-babylm/$SOURCE/ckpt.pt"
HFDIR="$HF_ROOT/$VARIANT"
case "$SOURCE" in
  *bl100m-*) TOK="$NANO_REPO/data/babylm_100m/tokenizer/bpe-16000.json" ;;
  *)         TOK="$NANO_REPO/data/babylm/tokenizer/bpe-16000.json" ;;
esac
RESULT_DIR="$EVAL_REPO/results/$VARIANT"

[[ -f "$CKPT" ]] || { echo "no checkpoint: $CKPT" >&2; exit 1; }
[[ ! -e "$HFDIR" ]] || { echo "HF export already exists: $HFDIR" >&2; exit 1; }
[[ ! -e "$RESULT_DIR" ]] || { echo "results already exist: $RESULT_DIR" >&2; exit 1; }

echo ">> bidirectional export $SOURCE -> $HFDIR"
"$PY" "$NANO_REPO/eval/convert_nanogpt_to_hf.py" \
  --ckpt "$CKPT" --tokenizer "$TOK" --out "$HFDIR" --bidirectional

export PATH="$(dirname "$PY"):$PATH"
cd "$EVAL_REPO"

echo ">> full zero-shot with shifted masked-next-token backend"
bash scripts/eval_zero_shot.sh "$HFDIR" mntp

# A successful full 2026 zero-shot run currently produces 16 result files.
N_RESULTS="$(find "$RESULT_DIR" -type f | wc -l | tr -d ' ')"
[[ "$N_RESULTS" -eq 16 ]] || {
  echo "expected 16 zero-shot files for $VARIANT, found $N_RESULTS" >&2
  exit 1
}

echo ">> bidirectional GLUE"
bash "$NANO_REPO/eval/eval_glue_variant.sh" "$VARIANT"

touch "$NANO_REPO/results/${VARIANT}.eval.done"
echo ">> complete: $VARIANT"
