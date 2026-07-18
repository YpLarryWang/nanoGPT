#!/usr/bin/env python3
"""Tokenize cleaned official BabyLM test files with a track tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

try:
    from .constants import EOT, SOURCES
    from .fetch_offdev import sha256_file
    from .fetch_offtest import PROTOCOL as SOURCE_PROTOCOL, TEST_RELEASE
except ImportError:  # direct script execution
    from constants import EOT, SOURCES
    from fetch_offdev import sha256_file
    from fetch_offtest import PROTOCOL as SOURCE_PROTOCOL, TEST_RELEASE


PROTOCOL = "official-test-tokenized-v1"
DTYPE = np.uint16


def load_source_manifest(data_dir: Path) -> dict:
    path = data_dir / "test_source_manifest.json"
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError(f"unexpected test source protocol: {manifest.get('protocol')!r}")
    if manifest.get("repo") != TEST_RELEASE["repo"]:
        raise ValueError(f"unexpected test repository: {manifest.get('repo')!r}")
    if manifest.get("revision") != TEST_RELEASE["revision"]:
        raise ValueError(f"unexpected test revision: {manifest.get('revision')!r}")
    records = {record["source"]: record for record in manifest.get("files", [])}
    if set(records) != set(SOURCES):
        raise ValueError("test source manifest does not contain exactly the six official sources")
    for source in SOURCES:
        raw_path = data_dir / "raw" / "test" / f"{source}.test"
        if sha256_file(raw_path) != records[source]["sha256"]:
            raise ValueError(f"raw test SHA-256 mismatch: {raw_path}")
    return manifest


def iter_token_ids(data_dir: Path, tokenizer: Tokenizer, eot_id: int):
    previous = None
    for source in SOURCES:
        path = data_dir / "clean" / "test" / f"{source}.txt"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                for token_id in tokenizer.encode(line.rstrip("\n")).ids:
                    if token_id == eot_id and previous == eot_id:
                        continue
                    yield token_id
                    previous = token_id
        if previous != eot_id:
            yield eot_id
            previous = eot_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", default="tokenizer/bpe-16000.json", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    tokenizer_path = args.tokenizer
    if not tokenizer_path.is_absolute():
        tokenizer_path = data_dir / tokenizer_path
    tokenizer_path = tokenizer_path.resolve()
    output_path = data_dir / "test.bin"
    manifest_path = data_dir / "test_manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing test outputs in {data_dir}")

    source_manifest = load_source_manifest(data_dir)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eot_id = tokenizer.token_to_id(EOT)
    if eot_id is None:
        raise ValueError(f"{EOT!r} is not registered in tokenizer")
    if tokenizer.get_vocab_size() > np.iinfo(DTYPE).max:
        raise ValueError("vocabulary does not fit in uint16")

    clean_inputs = []
    for source in SOURCES:
        path = data_dir / "clean" / "test" / f"{source}.txt"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty cleaned test source: {path}")
        clean_inputs.append(
            {"source": source, "path": str(path), "sha256": sha256_file(path)}
        )

    tokens = np.fromiter(iter_token_ids(data_dir, tokenizer, eot_id), dtype=DTYPE)
    if tokens.size < 2:
        raise ValueError("tokenized test set must contain at least two tokens")
    with output_path.open("xb") as handle:
        tokens.tofile(handle)

    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_protocol": source_manifest["protocol"],
        "repo": source_manifest["repo"],
        "revision": source_manifest["revision"],
        "source_manifest_sha256": sha256_file(data_dir / "test_source_manifest.json"),
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": sha256_file(tokenizer_path),
            "vocab_size": tokenizer.get_vocab_size(),
            "eot_id": eot_id,
        },
        "clean_inputs": clean_inputs,
        "bin": {
            "path": str(output_path),
            "tokens": int(tokens.size),
            "sha256": sha256_file(output_path),
            "dtype": "uint16",
        },
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"wrote {output_path}: {tokens.size:,} tokens, "
        f"sha256={payload['bin']['sha256']}",
        flush=True,
    )
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
