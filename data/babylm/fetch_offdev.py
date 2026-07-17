#!/usr/bin/env python3
"""Download pinned official BabyLM train/dev files for an offdev dataset.

The downloader is fail-closed: it refuses to overwrite an existing raw file,
checks the official whitespace-word count, and writes a SHA256 source manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

try:
    from .constants import RAW_FILENAME_TEMPLATES, SOURCES
except ImportError:  # direct script execution
    from constants import RAW_FILENAME_TEMPLATES, SOURCES


RELEASES = {
    "10m": {
        "repo": "BabyLM-community/BabyLM-2026-Strict-Small",
        "revision": "c92ab16b4f08858304b0815706065b3354d8fc0a",
        "words": {
            "bnc_spoken": 762_073,
            "childes": 2_841_101,
            "gutenberg": 2_557_721,
            "open_subtitles": 2_282_877,
            "simple_wiki": 1_531_437,
            "switchboard": 24_791,
        },
    },
    "100m": {
        "repo": "BabyLM-community/BabyLM-2026-Strict",
        "revision": "9e57baaaa91ac3c638746be14d1d5fa6c789f4cf",
        "words": {
            "bnc_spoken": 7_620_671,
            "childes": 28_410_878,
            "gutenberg": 25_576_896,
            "open_subtitles": 22_828_747,
            "simple_wiki": 15_314_317,
            "switchboard": 248_491,
        },
    },
}

DEV_RELEASE = {
    "repo": "BabyLM-community/BabyLM-dev",
    "revision": "169f42e32d0aaf65ec6b91d55bafad27a3afc729",
    "words": {
        "bnc_spoken": 1_252_593,
        "childes": 2_716_591,
        "gutenberg": 2_819_070,
        "open_subtitles": 2_077_019,
        "simple_wiki": 1_405_366,
        "switchboard": 148_340,
    },
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_counts(path: Path) -> tuple[int, int]:
    lines = words = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            lines += 1
            words += len(line.split())
    return lines, words


def download(url: str, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing raw file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    if part.exists():
        raise FileExistsError(f"remove or inspect incomplete download first: {part}")
    request = urllib.request.Request(url, headers={"User-Agent": "nanoGPT-BabyLM-offdev/1"})
    try:
        with urllib.request.urlopen(request) as response, part.open("xb") as out:
            while chunk := response.read(8 * 1024 * 1024):
                out.write(chunk)
        os.replace(part, destination)
    except BaseException:
        if part.exists():
            part.unlink()
        raise


def fetch_split(data_dir: Path, split: str, release: dict) -> list[dict]:
    records = []
    template = RAW_FILENAME_TEMPLATES[split]
    for source in SOURCES:
        filename = template.format(source=source)
        destination = data_dir / "raw" / split / filename
        url = (
            f"https://huggingface.co/datasets/{release['repo']}/resolve/"
            f"{release['revision']}/{filename}"
        )
        if destination.exists():
            print(f"verify existing {destination}", flush=True)
        else:
            print(f"download {url} -> {destination}", flush=True)
            download(url, destination)
        lines, words = text_counts(destination)
        expected = release["words"][source]
        if words != expected:
            raise ValueError(f"{destination}: {words:,} words, expected {expected:,}")
        records.append(
            {
                "source": source,
                "filename": filename,
                "url": url,
                "bytes": destination.stat().st_size,
                "lines": lines,
                "words": words,
                "sha256": sha256_file(destination),
            }
        )
        print(f"verified {source}: {lines:,} lines, {words:,} words", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=sorted(RELEASES), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if (data_dir / "source_manifest.json").exists():
        raise FileExistsError(f"dataset already has a source manifest: {data_dir}")
    train_release = RELEASES[args.track]
    payload = {
        "schema_version": 1,
        "protocol": "official-train-dev-v1",
        "track": args.track,
        "splits": {
            "train": {
                "repo": train_release["repo"],
                "revision": train_release["revision"],
                "files": fetch_split(data_dir, "train", train_release),
            },
            "dev": {
                "repo": DEV_RELEASE["repo"],
                "revision": DEV_RELEASE["revision"],
                "files": fetch_split(data_dir, "dev", DEV_RELEASE),
            },
        },
    }
    with (data_dir / "source_manifest.json").open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"wrote {data_dir / 'source_manifest.json'}")


if __name__ == "__main__":
    main()
