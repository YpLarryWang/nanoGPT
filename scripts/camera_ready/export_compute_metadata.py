#!/usr/bin/env python3
"""Export architecture facts for offline FLOP accounting; no GPU execution."""
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from model import GPT, GPTConfig

records = {}
for arm in ('baseline', 'static', 'dynamic'):
    cfg = GPTConfig(block_size=512, vocab_size=16000, n_layer=32, n_head=8,
                    n_embd=512, dropout=.1, bias=False, use_rmsnorm=True,
                    use_swiglu=True, use_rope=True, use_attn_gate=True,
                    use_attn_res=arm != 'baseline',
                    use_static_attn_res=arm == 'static', attn_res_block_size=8)
    with torch.device('meta'):
        model = GPT(cfg)
    sites = []
    if cfg.use_attn_res:
        completed = 0
        for i, block in enumerate(model.transformer.h):
            sites.append({'site': f'layer{i}.pre_attention', 'sources': completed + 1})
            if block.block_start:
                completed += 1
            sites.append({'site': f'layer{i}.pre_mlp', 'sources': completed + 1})
        sites.append({'site': 'final', 'sources': completed + 1})
    records[arm] = {
        'config': dataclasses.asdict(cfg),
        'unique_parameter_count': sum(p.numel() for p in model.parameters()),
        'parameter_shapes': {n: list(p.shape) for n, p in model.named_parameters()},
        'linear_shapes': {n: {'in': m.in_features, 'out': m.out_features,
                              'bias': m.bias is not None}
                          for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)},
        'embedding_head_tied': model.lm_head.weight is model.transformer.wte.weight,
        'actual_swiglu_hidden': model.transformer.h[0].mlp.c_gate.out_features,
        'routing_sites': sites,
        'sum_routing_sources_per_forward': sum(s['sources'] for s in sites),
    }
result = {
    'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'arms': records,
    'training': {'micro_batch': 8, 'sequence_length': 512, 'gradient_accumulation': 64,
                 'optimizer_updates': 471, 'tokens_per_update': 262144,
                 'total_training_tokens': 123469824, 'precision': 'bfloat16 autocast',
                 'optimizer': 'fused AdamW', 'gradient_checkpointing': False,
                 'compile': True},
    'counting_notes': [
        'These are structural facts, not measured FLOPs.',
        'State whether multiply-add counts as two FLOPs, and whether causal attention uses full-square or triangular counting.',
        'Training includes forward and backward; account separately for optimizer, validation, and evaluation if included.',
        'nanoGPT targets=None computes the vocabulary head only at the final position; training and full-token scoring compute it at all positions.',
        'Specify inference workload, sequence lengths, and KV-cache policy before quoting inference FLOPs.',
        'Do not use estimate_mfu 6N+attention as an exact routing-aware count.',
        'Tensor stacking and memory traffic can cost wall time without many floating-point operations.',
    ],
}
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + '\n')
print('Saved', out)
