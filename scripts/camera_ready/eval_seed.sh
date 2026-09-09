#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
NANO=$PWD
NAME=${1:?run name}; GPU=${2:?GPU}; PHASE=${3:-priority}
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4
PY=/workspace/envs/eval/bin/python
TRAIN_PY=/workspace/envs/train/bin/python
export PATH="$(dirname "$PY"):$PATH"
EVAL=/workspace/babylm-eval/strict
CKPT=$("$TRAIN_PY" - "$NAME" <<'PY'
import json,sys
from pathlib import Path
p=Path('out-babylm')/sys.argv[1];d=json.loads((p/'checkpoint_manifest.json').read_text());print((p/d['roles']['final']).resolve())
PY
)
OUT="$NANO/results/camera-ready/$NAME"
HF="/workspace/hf-models/$NAME"
EVAL_NAME=${NAME//./p}
HF_EVAL="/workspace/hf-models/$EVAL_NAME"
mkdir -p "$OUT"
# Permit early evaluation on a free GPU without overlapping the main queue.
exec 8>"$NANO/results/camera-ready/eval-gpu-${GPU}.lock"
flock 8
exec 9>"$OUT/${PHASE}.lock"
flock 9
if [[ "$PHASE" == priority ]]; then
 if [[ -f "$OUT/priority.done" && -s "$EVAL/results/$EVAL_NAME/main/zero_shot/causal/blimp/blimp_filtered/best_temperature_report.txt" ]]; then
  "$PY" - "$OUT" "$EVAL" "$EVAL_NAME" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0, 'eval')
from sync_eval_results import parse_average_accuracy
for split in ('dev','test'):
 d=json.loads((Path(sys.argv[1])/f'full_{split}.json').read_text())
 assert d['mean_nll'] > 0 and d['perplexity'] > 0
p=Path(sys.argv[2])/'results'/sys.argv[3]/'main/zero_shot/causal/blimp/blimp_filtered/best_temperature_report.txt'
assert 0 <= parse_average_accuracy(p) <= 100
PY
  echo "Reusing verified priority results: $NAME"
  exit 0
 fi
 for SPLIT in dev test; do
  if [[ ! -s "$OUT/full_$SPLIT.json" ]]; then
   "$TRAIN_PY" eval/full_dev_loss.py --checkpoint "$CKPT" --data-dir data/babylm_officialdev \
    --split "$SPLIT" --batch-size 32 --output-json "$OUT/full_$SPLIT.json" > "$OUT/full_$SPLIT.log" 2>&1
  fi
 done
 if [[ ! -s "$HF/config.json" ]]; then
  "$PY" eval/convert_nanogpt_to_hf.py --ckpt "$CKPT" --tokenizer data/babylm_officialdev/tokenizer/bpe-16000.json --out "$HF"
 fi
 if [[ ! -e "$HF_EVAL" ]]; then ln -s "$HF" "$HF_EVAL"; fi
 cd "$EVAL"
 "$PY" -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$HF_EVAL" --backend causal \
  --task blimp --data_path evaluation_data/full_eval/blimp_filtered --save_predictions --revision_name main
 touch "$OUT/priority.done"
elif [[ "$PHASE" == remaining ]]; then
 if [[ ! -e "$HF_EVAL" ]]; then ln -s "$HF" "$HF_EVAL"; fi
 cd "$EVAL"
 for PAIR in blimp:supplement_filtered ewok:ewok_filtered entity_tracking:entity_tracking comps:comps global_piqa_parallel:global_piqa_parallel global_piqa_nonparallel:global_piqa_nonparallel; do
  "$PY" -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$HF_EVAL" --backend causal \
   --task "${PAIR%:*}" --data_path "evaluation_data/full_eval/${PAIR#*:}" --save_predictions --revision_name main
 done
 touch "$OUT/remaining.done"
else
 echo "unknown phase: $PHASE" >&2; exit 2
fi
