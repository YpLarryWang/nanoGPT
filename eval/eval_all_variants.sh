#!/usr/bin/env bash
# Convert + evaluate every trained ablation variant, so the architecture ablation
# can be compared on BabyLM benchmarks (not just val loss).
#
#   bash eval/eval_all_variants.sh            # fast zero-shot on every variant (default)
#   bash eval/eval_all_variants.sh --full     # full zero-shot
#   bash eval/eval_all_variants.sh --full --glue
#
# Continues past any single-variant failure.
set -uo pipefail
: "${DATA:=/media/volume/yupei-data}"
: "${NANO_REPO:=$DATA/repo/nanoGPT}"

MODE="--fast"; EXTRA=""
for a in "$@"; do
  case "$a" in
    --full) MODE="" ;;
    --fast) MODE="--fast" ;;
    --glue) EXTRA="--glue" ;;
    *) echo "unknown arg: $a" >&2; exit 1 ;;
  esac
done

for ckpt in "$NANO_REPO"/out-babylm/*/ckpt.pt; do
  v="$(basename "$(dirname "$ckpt")")"
  [ "$v" = smoke ] && continue
  echo "==================== $v ($MODE $EXTRA) ===================="
  bash "$NANO_REPO/eval/eval_variant.sh" "$v" $MODE $EXTRA || echo "!! FAILED: $v"
done
echo "==== all variants done ===="
