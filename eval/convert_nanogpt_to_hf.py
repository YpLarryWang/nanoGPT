#!/usr/bin/env python
"""Convert a nanoGPT checkpoint (ckpt.pt) into a HuggingFace model directory that the
BabyLM 2026 evaluation pipeline can load with trust_remote_code=True.

The output dir contains: config.json (with auto_map), model.safetensors, a copy of the
self-contained modeling_nanogpt.py, and the tokenizer files. Works for every ablation
variant (LayerNorm/RMSNorm x MLP/SwiGLU x learned/RoPE).

Example:
  python eval/convert_nanogpt_to_hf.py \
      --ckpt out-babylm/bl100m-ln-mlp-learned/ckpt.pt \
      --tokenizer data/babylm_100m/tokenizer/bpe-16000.json \
      --out hf-models/bl100m-ln-mlp-learned
"""
import argparse
import os
import shutil
import sys

import torch

HF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_nanogpt")
sys.path.insert(0, HF_DIR)
from modeling_nanogpt import NanoGPTConfig, NanoGPTForCausalLM, NanoGPTModel  # noqa: E402

from transformers import PreTrainedTokenizerFast  # noqa: E402

EOT = "<|endoftext|>"
AUTO_MAP = {
    "AutoConfig": "modeling_nanogpt.NanoGPTConfig",
    "AutoModel": "modeling_nanogpt.NanoGPTModel",
    "AutoModelForCausalLM": "modeling_nanogpt.NanoGPTForCausalLM",
}


def _clean(k):
    return k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to nanoGPT ckpt.pt")
    ap.add_argument("--tokenizer", required=True, help="path to the bpe-*.json tokenizers file")
    ap.add_argument("--out", required=True, help="output HF model directory")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    margs = dict(ckpt["model_args"])
    state_dict = {_clean(k): v for k, v in ckpt["model"].items()}

    config = NanoGPTConfig(
        vocab_size=margs["vocab_size"],
        block_size=margs["block_size"],
        n_layer=margs["n_layer"],
        n_head=margs["n_head"],
        n_embd=margs["n_embd"],
        dropout=0.0,
        bias=margs["bias"],
        use_rmsnorm=margs["use_rmsnorm"],
        use_swiglu=margs["use_swiglu"],
        swiglu_mult=margs.get("swiglu_mult", 8 / 3),
        use_rope=margs["use_rope"],
        tie_word_embeddings=True,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        architectures=["NanoGPTForCausalLM"],
    )
    config.auto_map = AUTO_MAP

    model = NanoGPTForCausalLM(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]  # tied to wte; fine
    assert not missing, f"missing keys after load: {missing}"
    assert not unexpected, f"unexpected keys after load: {unexpected}"

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    model = model.to(dtype).eval()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    shutil.copy(os.path.join(HF_DIR, "modeling_nanogpt.py"), os.path.join(args.out, "modeling_nanogpt.py"))

    tok = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer, eos_token=EOT, pad_token=EOT)
    eot_id = tok.convert_tokens_to_ids(EOT)
    assert eot_id == 0, f"expected {EOT} id 0, got {eot_id}"
    if len(tok) != margs["vocab_size"]:
        print(f"WARNING: tokenizer size {len(tok)} != model vocab {margs['vocab_size']}")
    tok.save_pretrained(args.out)

    print(
        f"[ok] {args.out}  ({config.n_layer}L/{config.n_head}H/{config.n_embd}d "
        f"rms={config.use_rmsnorm} swiglu={config.use_swiglu} rope={config.use_rope} "
        f"vocab={config.vocab_size} dtype={args.dtype})"
    )


if __name__ == "__main__":
    main()
