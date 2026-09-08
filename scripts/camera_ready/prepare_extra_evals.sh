#!/usr/bin/env bash
set -euo pipefail
PY=/workspace/envs/eval/bin/python
cd /workspace/babylm-eval/strict
"$PY" -m evaluation_pipeline.global_piqa.dl
"$PY" -c 'import nltk; nltk.download("punkt",quiet=True); nltk.download("punkt_tab",quiet=True)'
"$PY" -m evaluation_pipeline.ewok.dl_and_filter
"$PY" - <<'PY'
import json
from pathlib import Path
from huggingface_hub import HfApi
for task,n in [('global_piqa_parallel',103),('global_piqa_nonparallel',100)]:
 p=Path('evaluation_data/full_eval')/task/'eng_latn.jsonl'
 rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
 assert len(rows)==n,(task,len(rows),n)
 print(task,len(rows))
sources={repo:HfApi().repo_info(repo,repo_type='dataset').sha for repo in ['mrlbenchmarks/global-piqa-parallel','mrlbenchmarks/global-piqa-nonparallel','ewok-core/ewok-core-1.0']}
Path('camera_ready_extra_eval_sources.json').write_text(json.dumps({'resolved_heads_after_upstream_download':sources},indent=2))
PY
