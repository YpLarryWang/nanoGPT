#!/usr/bin/env python3
"""Download and verify the pinned official BabyLM 2026 test set.

The six files come directly from ``BabyLM-community/BabyLM-Test``.  This
downloader is deliberately fail-closed: it pins the repository commit and
checks byte size, whitespace-word count, and SHA-256 before writing a source
manifest.  Existing files are verified but never overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .constants import SOURCES
    from .fetch_offdev import download, sha256_file, text_counts
except ImportError:  # direct script execution
    from constants import SOURCES
    from fetch_offdev import download, sha256_file, text_counts


PROTOCOL = "official-test-v1"
TEST_RELEASE = {
    "repo": "BabyLM-community/BabyLM-Test",
    "revision": "2c47b98e2dc3707465aed81da69dc36cdca5d13b",
    "files": {
        "bnc_spoken": {
            "bytes": 4_889_225,
            "words": 932_334,
            "sha256": "bdfc15bb70e1bd5f66d6a6113ef47995b7fe7fac247bd8afddfac87b86627b4b",
        },
        "childes": {
            "bytes": 14_707_424,
            "words": 2_700_128,
            "sha256": "c6b9e5d5479d003381a9a5ea8905ed82ac9c418cbc53395d4234bbb218ee8424",
        },
        "gutenberg": {
            "bytes": 13_296_106,
            "words": 2_404_516,
            "sha256": "df309abec59a9143cb5220d81438297d24526024608e6bb795ca8c1a7069c9bd",
        },
        "open_subtitles": {
            "bytes": 10_388_196,
            "words": 1_949_898,
            "sha256": "59cf3e496921fb89f2581cae4f91ce5d2009513ad10778eb4181381d7ea44066",
        },
        "simple_wiki": {
            "bytes": 7_618_762,
            "words": 1_300_077,
            "sha256": "e3b1dec6f381115c82630b229ba427edd0f79553ccb9b40fdfc965e31bc7da70",
        },
        "switchboard": {
            "bytes": 823_158,
            "words": 167_133,
            "sha256": "f30501f856a0f63571f81716b2e7f7420cfc9ca4b4705671111063065792e713",
        },
    },
}


def verify_file(path: Path, expected: dict) -> dict:
    lines, words = text_counts(path)
    actual = {
        "bytes": path.stat().st_size,
        "lines": lines,
        "words": words,
        "sha256": sha256_file(path),
    }
    for key in ("bytes", "words", "sha256"):
        if actual[key] != expected[key]:
            raise ValueError(
                f"{path}: {key}={actual[key]!r}, expected {expected[key]!r}"
            )
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    manifest_path = data_dir / "test_source_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"test source manifest already exists: {manifest_path}")

    records = []
    for source in SOURCES:
        filename = f"{source}.test"
        destination = data_dir / "raw" / "test" / filename
        url = (
            f"https://huggingface.co/datasets/{TEST_RELEASE['repo']}/resolve/"
            f"{TEST_RELEASE['revision']}/{filename}"
        )
        if destination.exists():
            print(f"verify existing {destination}", flush=True)
        else:
            print(f"download {url} -> {destination}", flush=True)
            download(url, destination)
        actual = verify_file(destination, TEST_RELEASE["files"][source])
        records.append({"source": source, "filename": filename, "url": url, **actual})
        print(
            f"verified {source}: {actual['lines']:,} lines, "
            f"{actual['words']:,} words, sha256={actual['sha256']}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "repo": TEST_RELEASE["repo"],
        "revision": TEST_RELEASE["revision"],
        "files": records,
        "total_bytes": sum(record["bytes"] for record in records),
        "total_lines": sum(record["lines"] for record in records),
        "total_words": sum(record["words"] for record in records),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
