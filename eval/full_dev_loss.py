#!/usr/bin/env python3
"""Deterministic full-official-dev next-token loss for native nanoGPT checkpoints.

The protocol follows the token-weighted NLL accounting used by Hugging Face's
fixed-context perplexity example, adapted to this repository's native forward:
``GPT(x, y)`` expects targets that are already shifted by one token.  Windows
therefore start at 0, block_size, 2*block_size, ... and read one extra token;
every token in val.bin except the first is scored exactly once.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model import GPT, GPTConfig  # noqa: E402


PROTOCOL = "full-dev-next-token-v1"
DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def token_batches(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield adjacent x/y blocks that score data[1:] exactly once."""
    if block_size <= 0 or batch_size <= 0:
        raise ValueError("block_size and batch_size must be positive")
    prediction_tokens = len(data) - 1
    if prediction_tokens <= 0:
        raise ValueError("val.bin must contain at least two tokens")

    full_windows, remainder = divmod(prediction_tokens, block_size)
    for first_window in range(0, full_windows, batch_size):
        count = min(batch_size, full_windows - first_window)
        starts = ((first_window + offset) * block_size for offset in range(count))
        x = np.stack(
            [np.asarray(data[start:start + block_size], dtype=np.int64) for start in starts]
        )
        starts = ((first_window + offset) * block_size for offset in range(count))
        y = np.stack(
            [np.asarray(data[start + 1:start + block_size + 1], dtype=np.int64) for start in starts]
        )
        yield torch.from_numpy(x), torch.from_numpy(y)

    if remainder:
        start = full_windows * block_size
        x = np.asarray(data[start:start + remainder], dtype=np.int64).copy()[None, :]
        y = np.asarray(data[start + 1:start + remainder + 1], dtype=np.int64).copy()[None, :]
        yield torch.from_numpy(x), torch.from_numpy(y)


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    log_every: int = 25,
) -> tuple[float, int, float]:
    """Return token-weighted mean NLL, scored-token count, and elapsed seconds."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    started = time.monotonic()
    device_type = device.type
    use_autocast = device_type == "cuda" and autocast_dtype is not None

    for batch_index, (x_cpu, y_cpu) in enumerate(
        token_batches(data, block_size, batch_size), start=1
    ):
        if device_type == "cuda":
            x_cpu = x_cpu.pin_memory()
            y_cpu = y_cpu.pin_memory()
        x = x_cpu.to(device, non_blocking=device_type == "cuda")
        y = y_cpu.to(device, non_blocking=device_type == "cuda")
        autocast = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if use_autocast
            else contextlib.nullcontext()
        )
        with autocast:
            _, mean_loss = model(x, y)
        if mean_loss is None or not torch.isfinite(mean_loss):
            raise RuntimeError(f"non-finite loss at batch {batch_index}: {mean_loss}")
        valid_tokens = y.numel()
        # Like the HF reference implementation, convert the mean CE back to NLL
        # using the exact number of valid targets, then aggregate in float64/Python.
        total_nll += float(mean_loss.detach().double().item()) * valid_tokens
        total_tokens += valid_tokens

        if log_every > 0 and batch_index % log_every == 0:
            elapsed = time.monotonic() - started
            rate = total_tokens / elapsed if elapsed else 0.0
            target = len(data) - 1
            print(
                f"progress tokens={total_tokens}/{target} "
                f"({100.0 * total_tokens / target:.2f}%) "
                f"loss={total_nll / total_tokens:.8f} tok/s={rate:.0f}",
                flush=True,
            )

    elapsed = time.monotonic() - started
    expected_tokens = len(data) - 1
    if total_tokens != expected_tokens:
        raise RuntimeError(f"scored {total_tokens} tokens, expected {expected_tokens}")
    return total_nll / total_tokens, total_tokens, elapsed


def resolve_dtype(requested: str, checkpoint: dict) -> tuple[str, torch.dtype | None]:
    name = checkpoint.get("config", {}).get("dtype", "float32") if requested == "auto" else requested
    if name not in DTYPES:
        raise ValueError(f"unsupported dtype {name!r}; choose one of {sorted(DTYPES)}")
    return name, None if name == "float32" else DTYPES[name]


def validate_data(checkpoint: dict, data_dir: Path, val_path: Path) -> tuple[str, str]:
    checkpoint_dataset = checkpoint.get("config", {}).get("dataset")
    if checkpoint_dataset and data_dir.name != checkpoint_dataset:
        raise ValueError(
            f"checkpoint dataset={checkpoint_dataset!r} does not match data directory {data_dir.name!r}"
        )
    expected = checkpoint.get("provenance", {}).get("data_fingerprints", {}).get(
        "val_bin_sha256"
    )
    if not expected:
        raise ValueError("checkpoint is missing provenance.data_fingerprints.val_bin_sha256")
    actual = sha256_file(val_path)
    if actual != expected:
        raise ValueError(f"val.bin SHA-256 mismatch: checkpoint={expected}, actual={actual}")
    return expected, actual


def write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="default: checkpoint config eval_batch_size")
    parser.add_argument("--dtype", choices=["auto", *DTYPES], default="auto")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    data_dir = args.data_dir.resolve()
    val_path = data_dir / "val.bin"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not val_path.is_file():
        raise FileNotFoundError(val_path)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    print(f"loading checkpoint={checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = checkpoint.get("model_args")
    state_dict = checkpoint.get("model")
    if not isinstance(model_args, dict) or not isinstance(state_dict, dict):
        raise ValueError("checkpoint must contain model_args and model state dictionaries")
    block_size = int(model_args["block_size"])
    checkpoint_eval_batch = checkpoint.get("config", {}).get("eval_batch_size")
    batch_size = args.batch_size if args.batch_size is not None else checkpoint_eval_batch
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("a positive --batch-size is required when checkpoint lacks eval_batch_size")
    dtype_name, autocast_dtype = resolve_dtype(args.dtype, checkpoint)
    expected_sha, actual_sha = validate_data(checkpoint, data_dir, val_path)

    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    if int(data.max()) >= int(model_args["vocab_size"]):
        raise ValueError("val.bin contains a token outside the checkpoint vocabulary")
    model = GPT(GPTConfig(**model_args))
    model.load_state_dict(clean_state_dict(state_dict))
    del checkpoint["model"]
    del state_dict
    model.to(device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(
        f"protocol={PROTOCOL} val_tokens={len(data)} scored_tokens={len(data) - 1} "
        f"block_size={block_size} batch_size={batch_size} dtype={dtype_name} "
        f"val_sha256={actual_sha}",
        flush=True,
    )
    avg_nll, scored_tokens, elapsed = evaluate_model(
        model=model,
        data=data,
        block_size=block_size,
        batch_size=batch_size,
        device=device,
        autocast_dtype=autocast_dtype,
        log_every=args.log_every,
    )
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "run_name": checkpoint.get("config", {}).get("wandb_run_name"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_role": checkpoint.get("checkpoint_role"),
        "iter_num": checkpoint.get("iter_num"),
        "dataset": checkpoint.get("config", {}).get("dataset"),
        "data_dir": str(data_dir),
        "val_bin_sha256_expected": expected_sha,
        "val_bin_sha256_actual": actual_sha,
        "val_tokens": int(len(data)),
        "scored_tokens": scored_tokens,
        "block_size": block_size,
        "batch_size": batch_size,
        "checkpoint_eval_batch_size": checkpoint_eval_batch,
        "dtype": dtype_name,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "mean_nll": avg_nll,
        "perplexity": math.exp(avg_nll),
        "elapsed_seconds": elapsed,
        "tokens_per_second": scored_tokens / elapsed if elapsed else None,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.output_json:
        write_json_exclusive(args.output_json.resolve(), result)
        print(f"wrote {args.output_json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
