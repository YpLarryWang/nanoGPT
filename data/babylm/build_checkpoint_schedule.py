#!/usr/bin/env python3
"""Build dual actual-word and BPE-token checkpoint schedules.

This replays the exact ``ShuffleSchedule`` used by train.py. Whitespace words
are aligned to the already-tokenized train.bin stream and counted when the
first BPE token of a word is present in a sampled training window. The script
refuses to continue unless BPE word-start marks exactly match ``str.split()``
on the clean training text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

EOT = "<|endoftext|>"
SOURCES = ["bnc_spoken", "childes", "gutenberg",
           "open_subtitles", "simple_wiki", "switchboard"]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def diagnose_word_start_mismatch(
    data_dir: Path,
    tokenizer: Tokenizer,
    lookup: np.ndarray,
    *,
    batch_size: int = 10_000,
) -> None:
    """Print the first clean line where BPE-space markers disagree with split()."""
    for source in SOURCES:
        path = data_dir / "clean" / "train" / f"{source}.txt"
        texts: list[str] = []
        line_numbers: list[int] = []

        def check_batch() -> bool:
            if not texts:
                return False
            for line_number, text, encoding in zip(
                line_numbers, texts, tokenizer.encode_batch(texts)
            ):
                expected = 0 if text == EOT else len(text.split())
                actual = int(lookup[np.asarray(encoding.ids, dtype=np.int64)].sum())
                if actual != expected:
                    token_strings = [tokenizer.id_to_token(token_id) for token_id in encoding.ids]
                    print(
                        "word-start mismatch: "
                        f"{path}:{line_number} split={expected} bpe_starts={actual} "
                        f"text={text!r} tokens={token_strings!r}"
                    )
                    return True
            return False

        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                texts.append(line.rstrip("\n"))
                line_numbers.append(line_number)
                if len(texts) == batch_size:
                    if check_batch():
                        return
                    texts.clear()
                    line_numbers.clear()
        if check_batch():
            return
    print("no per-line mismatch found; inspect source-boundary handling")


def build_word_starts(data_dir: Path, tokenizer_path: Path, output: Path) -> np.memmap:
    """Mark first BPE token of every whitespace word directly in train.bin.

    The tokenizer was trained with ByteLevel(add_prefix_space=True). Cleaning
    normalizes whitespace, so every whitespace word begins with a token whose
    serialized byte-level spelling starts with ``Ġ`` (the GPT-2 space byte).
    """
    train_bin = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eot_id = tokenizer.token_to_id(EOT)
    if eot_id is None:
        raise ValueError(f"{EOT!r} is not registered in {tokenizer_path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    marks_out = np.memmap(output, dtype=np.uint8, mode="w+", shape=(len(train_bin),))
    vocab = tokenizer.get_vocab()
    lookup = np.zeros(max(vocab.values()) + 1, dtype=np.uint8)
    for token, token_id in vocab.items():
        lookup[token_id] = int(token.startswith("Ġ"))
    lookup[eot_id] = 0
    if int(train_bin.max()) >= len(lookup):
        raise ValueError("train.bin contains an id outside the tokenizer vocabulary")
    chunk_size = 16 * 1024 * 1024
    for start in range(0, len(train_bin), chunk_size):
        stop = min(start + chunk_size, len(train_bin))
        marks_out[start:stop] = lookup[train_bin[start:stop]]
    marks_out.flush()

    clean_words = 0
    for source in SOURCES:
        path = data_dir / "clean" / "train" / f"{source}.txt"
        with path.open(encoding="utf-8") as f:
            for line in f:
                text = line.rstrip("\n")
                if text and text != EOT:
                    clean_words += len(text.split())
    marked_words = int(marks_out.sum(dtype=np.uint64))
    if marked_words != clean_words:
        diagnose_word_start_mismatch(data_dir, tokenizer, lookup)
        raise ValueError(
            "word-start/tokenizer invariant failed: "
            f"train.bin marks {marked_words:,} words but clean text has {clean_words:,}"
        )
    return marks_out


def shuffle_starts(
    *,
    data_len: int,
    block_size: int,
    batch_size: int,
    world_size: int,
    local_grad_accum: int,
    max_iters: int,
    seed: int,
) -> np.ndarray:
    """Byte-for-byte equivalent schedule construction to train.ShuffleSchedule."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    max_start = data_len - block_size - 1
    n_draws = (max_iters + 3) * local_grad_accum
    need = (n_draws * world_size + world_size) * batch_size
    parts: list[torch.Tensor] = []
    total = 0
    while total < need:
        offset = int(torch.randint(block_size, (1,), generator=generator))
        n_chunks = (max_start - offset) // block_size + 1
        starts = offset + block_size * torch.arange(n_chunks, dtype=torch.int64)
        parts.append(starts[torch.randperm(n_chunks, generator=generator)])
        total += n_chunks
    return torch.cat(parts).numpy()


def cumulative_exposure(
    starts: np.ndarray,
    word_starts: np.ndarray,
    *,
    block_size: int,
    batch_size: int,
    global_grad_accum: int,
    max_iters: int,
) -> list[int]:
    prefix = np.empty(len(word_starts) + 1, dtype=np.uint32)
    prefix[0] = 0
    np.cumsum(word_starts, dtype=np.uint32, out=prefix[1:])
    chunks_per_update = batch_size * global_grad_accum
    cumulative = np.zeros(max_iters + 1, dtype=np.uint64)
    for update in range(max_iters):
        batch_starts = starts[update * chunks_per_update : (update + 1) * chunks_per_update]
        if len(batch_starts) != chunks_per_update:
            raise ValueError("shuffle schedule exhausted while counting exposure")
        seen = prefix[batch_starts + block_size] - prefix[batch_starts]
        cumulative[update + 1] = cumulative[update] + seen.sum(dtype=np.uint64)
    return [int(x) for x in cumulative]


def standard_targets(final_count: int) -> list[int]:
    targets = [i * 1_000_000 for i in range(1, 11)]
    targets += [i * 10_000_000 for i in range(2, 11)]
    targets += [i * 100_000_000 for i in range(2, final_count // 100_000_000 + 1)]
    return [target for target in targets if target <= final_count]


def nearest_iter(cumulative: list[int], target: int) -> int:
    idx = int(np.searchsorted(cumulative, target, side="left"))
    if idx >= len(cumulative):
        return len(cumulative) - 1
    if idx > 0 and target - cumulative[idx - 1] <= cumulative[idx] - target:
        return idx - 1
    return idx


def million_label(value: int) -> str:
    millions = value / 1_000_000
    return f"{millions:g}M"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", default="tokenizer/bpe-16000.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--word-map", type=Path, help="reusable uint8 word-start map")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--global-grad-accum", type=int, default=16)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-iters", type=int, required=True)
    parser.add_argument("--sampler-seed", type=int, default=1337)
    args = parser.parse_args()

    if args.global_grad_accum % args.world_size:
        raise ValueError("global-grad-accum must be divisible by world-size")
    data_dir = args.data_dir.resolve()
    tokenizer_path = (data_dir / args.tokenizer).resolve()
    word_map_path = args.word_map or data_dir / "train.word_starts.uint8"
    if word_map_path.exists():
        word_starts = np.memmap(word_map_path, dtype=np.uint8, mode="r")
        train_len = os.path.getsize(data_dir / "train.bin") // np.dtype(np.uint16).itemsize
        if len(word_starts) != train_len:
            raise ValueError(f"word map has {len(word_starts)} entries; train.bin has {train_len}")
    else:
        word_starts = build_word_starts(data_dir, tokenizer_path, word_map_path)

    local_grad_accum = args.global_grad_accum // args.world_size
    starts = shuffle_starts(
        data_len=len(word_starts),
        block_size=args.block_size,
        batch_size=args.batch_size,
        world_size=args.world_size,
        local_grad_accum=local_grad_accum,
        max_iters=args.max_iters,
        seed=args.sampler_seed,
    )
    cumulative_words = cumulative_exposure(
        starts,
        word_starts,
        block_size=args.block_size,
        batch_size=args.batch_size,
        global_grad_accum=args.global_grad_accum,
        max_iters=args.max_iters,
    )
    tokens_per_iter = args.global_grad_accum * args.batch_size * args.block_size
    final_tokens = args.max_iters * tokens_per_iter

    by_iter: dict[int, list[dict]] = defaultdict(list)
    for series, cumulative, final_count in (
        ("words", cumulative_words, cumulative_words[-1]),
        ("tokens", [i * tokens_per_iter for i in range(args.max_iters + 1)], final_tokens),
    ):
        targets = standard_targets(final_count)
        for target in targets:
            it = nearest_iter(cumulative, target)
            by_iter[it].append(
                {
                    "series": series,
                    "name": f"{series}_{million_label(target)}",
                    "revision": f"chck_{million_label(target)}",
                    "target": target,
                    "actual": cumulative[it],
                }
            )
        if final_count not in targets:
            by_iter[args.max_iters].append(
                {
                    "series": series,
                    "name": f"{series}_final",
                    "revision": f"chck_{round(final_count / 1_000_000)}M",
                    "target": final_count,
                    "actual": final_count,
                }
            )

    checkpoints = []
    for it in sorted(by_iter):
        checkpoints.append(
            {
                "iter_num": it,
                "tokens_seen": it * tokens_per_iter,
                "words_seen": cumulative_words[it],
                "labels": by_iter[it],
            }
        )

    payload = {
        "schema_version": 1,
        "parameters": {
            "data_dir": str(data_dir),
            "tokenizer": str(Path(args.tokenizer)),
            "max_iters": args.max_iters,
            "block_size": args.block_size,
            "batch_size": args.batch_size,
            "global_grad_accum": args.global_grad_accum,
            "world_size": args.world_size,
            "tokens_per_iter": tokens_per_iter,
            "sampler_seed": args.sampler_seed,
        },
        "fingerprints": {
            "train_bin_sha256": sha256_file(data_dir / "train.bin"),
            "val_bin_sha256": (
                sha256_file(data_dir / "val.bin") if (data_dir / "val.bin").exists() else None
            ),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "word_map_sha256": sha256_file(word_map_path),
        },
        "counting": {
            "words": "whitespace words counted when their first BPE token occurs in a sampled x window",
            "tokens": "BPE tokens in sampled x windows",
            "checkpoint_rounding": "nearest completed optimizer update",
        },
        "cumulative_words": cumulative_words,
        "checkpoints": checkpoints,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(
        f"wrote {args.output}: {len(checkpoints)} unique checkpoints, "
        f"{sum(len(x['labels']) for x in checkpoints)} labels, "
        f"final={cumulative_words[-1]:,} words/{final_tokens:,} tokens"
    )


if __name__ == "__main__":
    main()
