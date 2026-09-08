#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/workspace/envs/train/bin/python}
D=data/babylm_officialdev
[[ -s "$D/tokenizer/bpe-16000.json" ]]
# The tokenizer is the original artifact, not trained again.
if [[ ! -s "$D/source_manifest.json" ]]; then
  "$PY" data/babylm/fetch_offdev.py --track 10m --data-dir "$D"
fi
for pair in train:train dev:val; do
  input=${pair%:*}; output=${pair#*:}
  if [[ ! -s "$D/clean/$output/bnc_spoken.txt" ]]; then
    "$PY" data/babylm/clean.py --input-split "$input" --raw-dir "$D/raw/$input" --out-dir "$D/clean/$output"
  fi
done
if [[ ! -s "$D/data_manifest.json" ]]; then "$PY" data/babylm/prepare.py --data-dir "$D"; fi
"$PY" - <<'PY'
import hashlib,json
from pathlib import Path
base=Path('data/babylm_officialdev')
f=json.loads(Path('config/checkpoint_schedules/bl10m-offdev-b8ga64-dual.json').read_text())['fingerprints']
for key,path in [('train_bin_sha256','train.bin'),('val_bin_sha256','val.bin'),('tokenizer_sha256','tokenizer/bpe-16000.json')]:
 actual=hashlib.sha256((base/path).read_bytes()).hexdigest()
 assert actual==f[key],(path,actual,f[key])
 print('MATCH',path,actual,flush=True)
(base/'camera_ready_train_data.verified').write_text(json.dumps(f,indent=2))
PY
if [[ ! -s "$D/test_source_manifest.json" ]]; then "$PY" data/babylm/fetch_offtest.py --data-dir "$D"; fi
if [[ ! -s "$D/clean/test/bnc_spoken.txt" ]]; then "$PY" data/babylm/clean.py --input-split test --raw-dir "$D/raw/test" --out-dir "$D/clean/test"; fi
if [[ ! -s "$D/test_manifest.json" ]]; then "$PY" data/babylm/prepare_test.py --data-dir "$D"; fi
touch "$D/camera_ready_data.done"
