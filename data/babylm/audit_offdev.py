#!/usr/bin/env python3
"""Audit an official-train/dev BabyLM dataset and derive its training budget."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .constants import EOT, RAW_FILENAME_TEMPLATES, SOURCES
except ImportError:  # direct script execution
    from constants import EOT, RAW_FILENAME_TEMPLATES, SOURCES


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_metrics(path: Path, *, ignore_eot: bool = False) -> dict:
    lines = words = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            text = line.rstrip("\n")
            lines += 1
            if not (ignore_eot and text == EOT):
                words += len(text.split())
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "lines": lines,
        "words": words,
        "sha256": sha256_file(path),
    }


def fingerprint(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def overlap_for_paths(train_paths: list[Path], dev_paths: list[Path]) -> dict:
    dev_all: set[bytes] = set()
    dev_long: set[bytes] = set()
    for dev_path in dev_paths:
        with dev_path.open(encoding="utf-8") as f:
            for line in f:
                text = line.rstrip("\n")
                if not text or text == EOT:
                    continue
                key = fingerprint(text)
                dev_all.add(key)
                if len(text) >= 40 and len(text.split()) >= 5:
                    dev_long.add(key)

    overlap_all: set[bytes] = set()
    overlap_long: set[bytes] = set()
    long_samples: list[str] = []
    for train_path in train_paths:
        with train_path.open(encoding="utf-8") as f:
            for line in f:
                text = line.rstrip("\n")
                if not text or text == EOT:
                    continue
                key = fingerprint(text)
                if key in dev_all:
                    overlap_all.add(key)
                if key in dev_long:
                    if key not in overlap_long and len(long_samples) < 10:
                        long_samples.append(text)
                    overlap_long.add(key)
    return {
        "dev_unique_non_eot_lines": len(dev_all),
        "exact_unique_line_overlaps": len(overlap_all),
        "dev_unique_long_lines": len(dev_long),
        "exact_unique_long_line_overlaps": len(overlap_long),
        "long_line_definition": "at least 5 whitespace words and 40 characters",
        "long_overlap_samples": long_samples,
    }


def overlap_for_source(train_path: Path, dev_path: Path) -> dict:
    return overlap_for_paths([train_path], [dev_path])


def verify_tokenizer_inputs(data_dir: Path) -> dict | None:
    path = data_dir / "tokenizer" / "tokenizer_manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    input_rows = manifest["tokenizer_inputs"]
    actual = {Path(item["path"]).resolve() for item in input_rows}
    expected = {
        (data_dir / "clean" / "train" / f"{source}.txt").resolve()
        for source in SOURCES
    }
    dev_paths = {
        (data_dir / "clean" / "val" / f"{source}.txt").resolve()
        for source in SOURCES
    }
    if actual != expected:
        raise ValueError(f"tokenizer inputs differ from complete clean/train set: {actual ^ expected}")
    if actual & dev_paths or manifest.get("dev_in_tokenizer_inputs") is not False:
        raise ValueError("dev data appears in tokenizer inputs")
    for item in input_rows:
        input_path = Path(item["path"])
        actual_sha = sha256_file(input_path)
        if actual_sha != item["sha256"]:
            raise ValueError(f"tokenizer input changed after training: {input_path}")
    return {
        "manifest": str(path.resolve()),
        "manifest_sha256": sha256_file(path),
        "train_only": True,
        "input_count": len(actual),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=512)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    report_path = args.report or data_dir / "audit_report.json"
    source_manifest_path = data_dir / "source_manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_records = {
        split_name: {row["source"]: row for row in split["files"]}
        for split_name, split in source_manifest["splits"].items()
    }

    splits = {}
    for raw_split, clean_split in (("train", "train"), ("dev", "val")):
        raw_rows = {}
        clean_rows = {}
        for source in SOURCES:
            raw_name = RAW_FILENAME_TEMPLATES[raw_split].format(source=source)
            raw_rows[source] = text_metrics(data_dir / "raw" / raw_split / raw_name)
            recorded = source_records[raw_split][source]
            for field in ("bytes", "lines", "words", "sha256"):
                if raw_rows[source][field] != recorded[field]:
                    raise ValueError(
                        f"raw {raw_split}/{source} {field} changed: "
                        f"{raw_rows[source][field]!r} != {recorded[field]!r}"
                    )
            clean_rows[source] = text_metrics(
                data_dir / "clean" / clean_split / f"{source}.txt", ignore_eot=True
            )
        splits[raw_split] = {
            "raw": raw_rows,
            "clean": clean_rows,
            "raw_total_words": sum(row["words"] for row in raw_rows.values()),
            "clean_total_words": sum(row["words"] for row in clean_rows.values()),
        }

    overlap = {
        source: overlap_for_source(
            data_dir / "clean" / "train" / f"{source}.txt",
            data_dir / "clean" / "val" / f"{source}.txt",
        )
        for source in SOURCES
    }
    overlap_totals = {
        key: sum(row[key] for row in overlap.values())
        for key in (
            "dev_unique_non_eot_lines",
            "exact_unique_line_overlaps",
            "dev_unique_long_lines",
            "exact_unique_long_line_overlaps",
        )
    }
    global_overlap = overlap_for_paths(
        [data_dir / "clean" / "train" / f"{source}.txt" for source in SOURCES],
        [data_dir / "clean" / "val" / f"{source}.txt" for source in SOURCES],
    )

    budget = None
    train_bin = data_dir / "train.bin"
    if train_bin.exists():
        if train_bin.stat().st_size % 2:
            raise ValueError(f"uint16 train.bin has odd byte size: {train_bin}")
        train_tokens = train_bin.stat().st_size // 2
        tokens_per_update = args.batch_size * args.grad_accum * args.block_size
        budget = {
            "source": "measured train.bin length",
            "train_tokens": train_tokens,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "block_size": args.block_size,
            "tokens_per_update": tokens_per_update,
            "derived_max_iters": round(args.epochs * train_tokens / tokens_per_update),
        }

    payload = {
        "schema_version": 1,
        "protocol": "official-train-dev-v1",
        "data_dir": str(data_dir),
        "source_manifest": {
            "path": str(source_manifest_path.resolve()),
            "sha256": sha256_file(source_manifest_path),
            "track": source_manifest["track"],
            "train_repo": source_manifest["splits"]["train"]["repo"],
            "train_revision": source_manifest["splits"]["train"]["revision"],
            "dev_repo": source_manifest["splits"]["dev"]["repo"],
            "dev_revision": source_manifest["splits"]["dev"]["revision"],
        },
        "splits": splits,
        "leakage_audit": {
            "method": "exact BLAKE2b-128 fingerprints of cleaned lines, reported per source",
            "per_source": overlap,
            "totals": overlap_totals,
            "global_across_sources": global_overlap,
            "note": "Short conversational duplicates are expected; inspect long overlaps separately.",
        },
        "tokenizer_provenance": verify_tokenizer_inputs(data_dir),
        "training_budget": budget,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "train_raw_words": splits["train"]["raw_total_words"],
        "train_clean_words": splits["train"]["clean_total_words"],
        "dev_raw_words": splits["dev"]["raw_total_words"],
        "dev_clean_words": splits["dev"]["clean_total_words"],
        "overlap_totals": overlap_totals,
        "global_overlap": global_overlap,
        "tokenizer_train_only": payload["tokenizer_provenance"],
        "training_budget": budget,
    }, indent=2))


if __name__ == "__main__":
    main()
