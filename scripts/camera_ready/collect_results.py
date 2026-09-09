#!/usr/bin/env python3
"""Collect endpoint evidence without modifying the author's scoreboards."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('sync', ROOT / 'eval/sync_eval_results.py')
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def collect(root, eval_root):
    records = []
    for seed in (1337, 1338, 1339):
        name = f'bl10m-d512L32-do0.1-gate-attnres8-static-offdev-endpoint-b8ga64-s{seed}'
        row = {'run_name': name, 'seed': seed, 'metrics': {}, 'sources': {}, 'missing': []}
        metrics = row['metrics']
        for split in ('dev', 'test'):
            path = root / name / f'full_{split}.json'
            if not path.exists():
                row['missing'].append(f'full_{split}')
                continue
            data = json.loads(path.read_text())
            value = data['mean_nll']
            if not math.isfinite(value):
                raise ValueError(f'nonfinite NLL: {path}')
            metrics[f'full_{split}_nll'] = value
            metrics[f'full_{split}_perplexity'] = data['perplexity']
            row['sources'][f'full_{split}'] = str(path)
        for key, parts in sync.ZERO_REPORTS_FULL.items():
            path = eval_root / name.replace('.', 'p') / 'main/zero_shot/causal' / parts[0] / parts[1] / 'best_temperature_report.txt'
            if not path.exists():
                row['missing'].append(key)
                continue
            value = sync.parse_average_accuracy(path)
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f'invalid percentage: {path}')
            metrics[key] = value
            row['sources'][key] = str(path)
        groups = {
            'avg5': ['blimp', 'supplement', 'ewok', 'entity_tracking', 'comps'],
            'reliable4': ['blimp', 'supplement', 'ewok', 'comps'],
            'global_piqa': ['global_piqa_parallel', 'global_piqa_nonparallel'],
        }
        for key, members in groups.items():
            if all(m in metrics for m in members):
                metrics[key] = sum(metrics[m] for m in members) / len(members)
        row['complete'] = not row['missing']
        records.append(row)
    return {'complete': all(r['complete'] for r in records), 'runs': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results/camera-ready')
    parser.add_argument('--eval-results', type=Path, default=Path('/workspace/babylm-eval/strict/results'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.results, args.eval_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))
