# BabyLM 2026 evaluation for nanoGPT ablations (strict track)

This directory bridges the nanoGPT ablation checkpoints to the official
[BabyLM 2026 evaluation pipeline](https://github.com/babylm-org/babylm-eval)
(`strict/`). It converts a nanoGPT `ckpt.pt` into a `trust_remote_code`
HuggingFace model and runs zero-shot + GLUE evaluation on it.

## What's here
- `hf_nanogpt/modeling_nanogpt.py` — **self-contained** HF wrapper for the nanoGPT
  architecture (LayerNorm/RMSNorm × MLP/SwiGLU × learned/RoPE). Defines
  `NanoGPTConfig`, `NanoGPTModel` (AutoModel backbone → used by GLUE), and
  `NanoGPTForCausalLM` (AutoModelForCausalLM → used by zero-shot). Parameter names
  mirror nanoGPT's `model.py`, so checkpoints load with no remapping. Differs from
  `model.py` in exactly two eval-required ways: it returns full-sequence logits and
  honours a padding `attention_mask`.
- `convert_nanogpt_to_hf.py` — `ckpt.pt` → HF dir (weights + config with `auto_map`
  + copied modeling file + tokenizer). Strips the `_orig_mod.` (torch.compile) prefix.
- `parity_check.py` — asserts the HF wrapper's logits equal nanoGPT's own forward
  (validated **max|diff| = 0.0**, bit-exact).
- `eval_variant.sh` — convert one variant + run the strict zero-shot suite (+`--glue`, +`--fast`).
- `eval_all_variants.sh` — do that for every trained variant (ablation-on-benchmarks).

## Where things live (jetstream `yupei-project-0415-g3large`, has an A100 20C)
- eval pipeline: `/media/volume/yupei-data/repo/babylm-eval` (cloned from GitHub)
- isolated venv: `/media/volume/yupei-data/envs/babylm-eval` (py3.12, torch 2.7+cu126,
  transformers 4.51.3 — **separate** from the training env `envs/nanogpt`, whose
  transformers 5.x would break the pipeline's pins)
- baselines: `/media/volume/yupei-data/models/babylm-baselines/{strict,strict-small}`
- converted models: `/media/volume/yupei-data/hf-models/<variant>`
- eval data: `.../babylm-eval/strict/evaluation_data/{full_eval,fast_eval}`

## Run it
```bash
# one variant, full zero-shot (blimp, supplement, ewok, entity_tracking, comps, reading)
bash eval/eval_variant.sh bl100m-rms-swiglu-rope
# fast subset (for checkpoint sweeps); also: --glue to add fine-tuning (slow)
bash eval/eval_variant.sh bl100m-rms-swiglu-rope --fast
# whole ablation
bash eval/eval_all_variants.sh --fast
```
Results land under `babylm-eval/strict/results/<variant>/`.

## Tasks (2026 strict track)
- **Zero-shot** (backend `causal`): BLiMP, BLiMP-supplement, EWoK, entity_tracking,
  COMPS, reading. Higher-prob-to-correct-sentence ranking.
- **GLUE fine-tuning**: boolq, multirc, rte, wsc, mrpc, qqp, mnli (10 epochs each).

Reference baseline (official, full eval): BLiMP 74.53 / supplement 65.00 / comps 55.85
/ entity_tracking 23.58 (strict) — our installed baseline reproduced BLiMP-fast at 75.03.

## Notes / gotchas
- **EWoK**: not in the auto-download. `python -m evaluation_pipeline.ewok.dl_and_filter`
  (needs `nltk.download('punkt','punkt_tab')`) fetches `full_eval/ewok_filtered`. The
  fast set `fast_eval/ewok_fast.zip` unzips with password `BabyLM2025`.
- **Tokenizer**: 100M variants use `data/babylm_100m/tokenizer/bpe-16000.json`,
  10M variants use `data/babylm/tokenizer/bpe-16000.json` (`<|endoftext|>` = id 0).
- **Full challenge submission** additionally needs per-word-budget checkpoints
  (1M..10M, 10M..100M[, ..1000M]) fast-evaluated + AoA, then
  `bash scripts/collate_preds.sh <model> causal strict --fast`. Requires saving those
  intermediate checkpoints during training (not yet wired into train.py).
