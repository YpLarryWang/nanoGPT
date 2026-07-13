#!/usr/bin/env bash
# Export one hybrid checkpoint with full bidirectional attention, evaluate its
# shifted masked-next-token objective with the official MNTP backend, then run
# the standard GLUE suite on the same bidirectional backbone.
#
# Usage:
#   bash eval/eval_bidirectional_variant.sh <causal-variant> [--role best|final] [--glue-only]

set -euo pipefail

SOURCE="${1:?usage: eval_bidirectional_variant.sh <causal-variant> [--role best|final] [--glue-only]}"; shift
ROLE=best
GLUE_ONLY=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --role) ROLE="${2:?--role needs best or final}"; shift ;;
    --glue-only) GLUE_ONLY=1 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
[[ "$ROLE" = best || "$ROLE" = final ]] || { echo "invalid role: $ROLE" >&2; exit 2; }
VARIANT="${SOURCE}-bidir"

: "${DATA:=/media/volume/yupei-data}"
: "${NANO_REPO:=$DATA/repo/nanoGPT}"
: "${EVAL_REPO:=$DATA/repo/babylm-eval/strict}"
: "${PY:=$DATA/envs/babylm-eval/bin/python}"
: "${HF_ROOT:=$DATA/hf-models}"

RUN_DIR="$NANO_REPO/out-babylm/$SOURCE"
MANIFEST="$RUN_DIR/checkpoint_manifest.json"
[[ -f "$MANIFEST" ]] || { echo "no checkpoint manifest: $MANIFEST" >&2; exit 1; }
CKPT="$($PY -c 'import json,os,sys; p=sys.argv[1]; role=sys.argv[2]; d=json.load(open(p)); print(os.path.join(os.path.dirname(p), d["roles"][role]))' "$MANIFEST" "$ROLE")"
HFDIR="$HF_ROOT/$VARIANT"
case "$SOURCE" in
  *bl100m-*) TOK="$NANO_REPO/data/babylm_100m/tokenizer/bpe-16000.json" ;;
  *)         TOK="$NANO_REPO/data/babylm/tokenizer/bpe-16000.json" ;;
esac
RESULT_DIR="$EVAL_REPO/results/$VARIANT"

[[ -f "$CKPT" ]] || { echo "no checkpoint: $CKPT" >&2; exit 1; }
[[ ! -e "$HFDIR" ]] || { echo "HF export already exists: $HFDIR" >&2; exit 1; }
[[ ! -e "$RESULT_DIR" ]] || { echo "results already exist: $RESULT_DIR" >&2; exit 1; }

echo ">> bidirectional export $SOURCE ($ROLE: $CKPT) -> $HFDIR"
"$PY" "$NANO_REPO/eval/convert_nanogpt_to_hf.py" \
  --ckpt "$CKPT" --tokenizer "$TOK" --out "$HFDIR" --bidirectional

export PATH="$(dirname "$PY"):$PATH"
cd "$EVAL_REPO"

if [[ "$GLUE_ONLY" = 0 ]]; then
  echo ">> full zero-shot with shifted masked-next-token backend"
  bash scripts/eval_zero_shot.sh "$HFDIR" mntp

  echo ">> GlobalPIQA"
  for task in global_piqa_parallel global_piqa_nonparallel; do
    "$PY" -m evaluation_pipeline.sentence_zero_shot.run \
      --model_path_or_name "$HFDIR" --backend mntp --task "$task" \
      --data_path "evaluation_data/full_eval/${task}" --save_predictions \
      --revision_name main
  done

  # Check named deliverables instead of a brittle total file count. Upstream can
  # add reports without invalidating a completed run.
  for required in \
    "$RESULT_DIR/main/zero_shot/mntp/blimp/blimp_filtered/best_temperature_report.txt" \
    "$RESULT_DIR/main/zero_shot/mntp/comps/comps/best_temperature_report.txt" \
    "$RESULT_DIR/main/zero_shot/mntp/global_piqa_parallel/global_piqa_parallel/best_temperature_report.txt" \
    "$RESULT_DIR/main/zero_shot/mntp/global_piqa_nonparallel/global_piqa_nonparallel/best_temperature_report.txt" \
    "$RESULT_DIR/main/zero_shot/mntp/reading/report.txt"; do
    [[ -f "$required" ]] || { echo "missing zero-shot deliverable: $required" >&2; exit 1; }
  done
fi

echo ">> bidirectional GLUE"
SYNC_METADATA_FROM="$SOURCE" bash "$NANO_REPO/eval/eval_glue_variant.sh" "$VARIANT"

touch "$NANO_REPO/results/${VARIANT}.eval.done"
echo ">> complete: $VARIANT"
if [[ "$GLUE_ONLY" = 0 ]]; then
  "$PY" "$NANO_REPO/eval/sync_eval_results.py" "$VARIANT" --full --backend mntp \
    --metadata-from "$SOURCE" --eval-repo "$EVAL_REPO"
fi
