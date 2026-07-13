#!/usr/bin/env bash
# Run full MNTP zero-shot evaluation for an existing bidirectional HF export.
#
# This is the companion to eval_bidirectional_variant.sh --glue-only.  It
# deliberately reuses the same <source>-bidir export and result directory, so
# zero-shot and GLUE are evaluated with the same weights and attention mode.
#
# Usage:
#   bash eval/eval_bidirectional_zero_shot.sh <source> [--role best|final]

set -euo pipefail

SOURCE="${1:?usage: eval_bidirectional_zero_shot.sh <source> [--role best|final]}"; shift
ROLE=best
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --role) ROLE="${2:?--role needs best or final}"; shift ;;
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
HFDIR="$HF_ROOT/$VARIANT"
RESULT_DIR="$EVAL_REPO/results/$VARIANT"
MNTP_DIR="$RESULT_DIR/main/zero_shot/mntp"

[[ -f "$MANIFEST" ]] || { echo "no checkpoint manifest: $MANIFEST" >&2; exit 1; }
CKPT="$($PY -c 'import json,os,sys; p=sys.argv[1]; role=sys.argv[2]; d=json.load(open(p)); print(os.path.join(os.path.dirname(p), d["roles"][role]))' "$MANIFEST" "$ROLE")"
[[ -f "$CKPT" ]] || { echo "no checkpoint: $CKPT" >&2; exit 1; }
[[ -d "$HFDIR" ]] || { echo "no existing bidirectional HF export: $HFDIR" >&2; exit 1; }
[[ -d "$RESULT_DIR" ]] || { echo "no existing bidirectional result directory: $RESULT_DIR" >&2; exit 1; }
[[ ! -e "$MNTP_DIR" ]] || { echo "MNTP results already exist: $MNTP_DIR" >&2; exit 1; }

# Refuse to score a causal export or an export made from a different role.
"$PY" - "$HFDIR/config.json" "$HFDIR/checkpoint_source.json" "$CKPT" <<'PY'
import json
import os
import sys

config_path, source_path, checkpoint_path = sys.argv[1:]
config = json.load(open(config_path))
source = json.load(open(source_path))
if config.get("bidirectional") is not True:
    raise SystemExit(f"HF export is not bidirectional: {config_path}")
if source.get("filename") != os.path.basename(checkpoint_path):
    raise SystemExit(
        f"HF export checkpoint {source.get('filename')!r} != requested "
        f"{os.path.basename(checkpoint_path)!r}"
    )
PY

export PATH="$(dirname "$PY"):$PATH"
cd "$EVAL_REPO"

echo ">> full bidirectional zero-shot: $VARIANT ($ROLE)"
bash scripts/eval_zero_shot.sh "$HFDIR" mntp

echo ">> GlobalPIQA (MNTP)"
for task in global_piqa_parallel global_piqa_nonparallel; do
  "$PY" -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name "$HFDIR" --backend mntp --task "$task" \
    --data_path "evaluation_data/full_eval/${task}" --save_predictions \
    --revision_name main
done

for required in \
  "$MNTP_DIR/blimp/blimp_filtered/best_temperature_report.txt" \
  "$MNTP_DIR/blimp/supplement_filtered/best_temperature_report.txt" \
  "$MNTP_DIR/ewok/ewok_filtered/best_temperature_report.txt" \
  "$MNTP_DIR/entity_tracking/entity_tracking/best_temperature_report.txt" \
  "$MNTP_DIR/comps/comps/best_temperature_report.txt" \
  "$MNTP_DIR/global_piqa_parallel/global_piqa_parallel/best_temperature_report.txt" \
  "$MNTP_DIR/global_piqa_nonparallel/global_piqa_nonparallel/best_temperature_report.txt" \
  "$MNTP_DIR/reading/report.txt"; do
  [[ -f "$required" ]] || { echo "missing zero-shot deliverable: $required" >&2; exit 1; }
done

SOURCE_NOTE="ours(measured;manifest-${ROLE};bidirectional-attention zero-shot;MNTP backend;entity-filter+GlobalPIQA;$(date +%F))"
"$PY" "$NANO_REPO/eval/sync_eval_results.py" "$VARIANT" --full --backend mntp \
  --metadata-from "$SOURCE" --eval-repo "$EVAL_REPO" --source-note "$SOURCE_NOTE"

touch "$NANO_REPO/results/${VARIANT}.zero-shot.done"
echo ">> complete: $VARIANT"
