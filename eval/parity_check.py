#!/usr/bin/env python
"""Sanity check: the converted HF model must produce logits identical to nanoGPT's
own model.py forward on the same (unpadded) token inputs.

  python eval/parity_check.py \
      --ckpt out-babylm/bl100m-ln-mlp-learned/ckpt.pt \
      --hf hf-models/bl100m-ln-mlp-learned
"""
import argparse
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # nanoGPT model.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_nanogpt"))

from model import GPT, GPTConfig  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402


def _clean(k):
    return k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--t", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    margs = ckpt["model_args"]
    gpt = GPT(GPTConfig(**margs))
    gpt.load_state_dict({_clean(k): v for k, v in ckpt["model"].items()})
    gpt.eval()

    hf = AutoModelForCausalLM.from_pretrained(args.hf, trust_remote_code=True).eval()

    torch.manual_seed(0)
    T = min(args.t, margs["block_size"])
    idx = torch.randint(0, margs["vocab_size"], (args.n, T))

    with torch.no_grad():
        nano_logits, _ = gpt(idx, targets=idx)  # targets -> full (B,T,V) logits
        hf_logits = hf(input_ids=idx).logits

    diff = (nano_logits.float() - hf_logits.float()).abs()
    print(f"shapes: nano={tuple(nano_logits.shape)} hf={tuple(hf_logits.shape)}")
    print(f"max|diff|={diff.max().item():.3e}  mean|diff|={diff.mean().item():.3e}  tol={args.tol}")
    ok = diff.max().item() < args.tol
    print("PARITY OK" if ok else "PARITY FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
