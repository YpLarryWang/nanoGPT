#!/usr/bin/env python3
"""Download only endpoint zero-shot resources; record resolved dataset revisions."""
import json,os
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
root=Path('/workspace/babylm-eval/strict');root.mkdir(parents=True,exist_ok=True)
repo='BabyLM-community/BabyLM-2026-Strict-Evals'
sha=HfApi().repo_info(repo,repo_type='dataset').sha
patterns=[f'evaluation_data/full_eval/{name}/**' for name in ['blimp_filtered','supplement_filtered','entity_tracking','comps','global_piqa_parallel','global_piqa_nonparallel','ewok_filtered']]
snapshot_download(repo_id=repo,repo_type='dataset',revision=sha,local_dir=root,allow_patterns=patterns)
(root/'camera_ready_eval_sources.json').write_text(json.dumps({'repo':repo,'revision':sha,'allow_patterns':patterns},indent=2))
for name in ['blimp_filtered','supplement_filtered','entity_tracking','comps']:
 p=root/'evaluation_data/full_eval'/name
 assert p.is_dir() and any(p.rglob('*.json*')),(name,'missing')
print('Core endpoint datasets downloaded; GlobalPIQA and EWoK use their dedicated upstream downloaders.',flush=True)
