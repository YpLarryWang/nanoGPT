#!/usr/bin/env python3
"""Convert and push one manifest series as Hugging Face revisions.

The same local run may contain both ``words`` and ``tokens`` labels. Choose one
at upload time; no retraining or checkpoint copying is required.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def selected_milestones(manifest: dict, series: str) -> list[tuple[dict, dict]]:
    selected = []
    revisions = set()
    for checkpoint in manifest.get("checkpoints", []):
        if checkpoint.get("role") != "milestone":
            continue
        for label in checkpoint.get("labels", []):
            if label.get("series") != series:
                continue
            revision = label["revision"]
            if revision in revisions:
                raise ValueError(f"duplicate revision in {series} series: {revision}")
            revisions.add(revision)
            selected.append((checkpoint, label))
    return sorted(selected, key=lambda pair: pair[0]["iter_num"])


def convert(python: str, converter: Path, checkpoint: Path, tokenizer: Path, output: Path, dtype: str):
    subprocess.run(
        [
            python,
            str(converter),
            "--ckpt", str(checkpoint),
            "--tokenizer", str(tokenizer),
            "--out", str(output),
            "--dtype", dtype,
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--series", required=True, choices=["words", "tokens"])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--python", default="python")
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path(__file__).with_name("convert_nanogpt_to_hf.py"),
    )
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    with (run_dir / "checkpoint_manifest.json").open(encoding="utf-8") as f:
        manifest = json.load(f)
    milestones = selected_milestones(manifest, args.series)
    if not milestones:
        raise SystemExit(f"manifest contains no {args.series!r} milestone checkpoints")

    final_relative = manifest.get("roles", {}).get("final")
    if not final_relative:
        raise SystemExit("manifest has no final checkpoint")
    final_checkpoint = run_dir / final_relative

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="babylm-hf-") as tmp:
        tmp_root = Path(tmp)
        converted_by_path: dict[Path, Path] = {}
        for checkpoint_entry, label in milestones:
            checkpoint = run_dir / checkpoint_entry["path"]
            output = converted_by_path.get(checkpoint)
            if output is None:
                output = tmp_root / f"iter-{checkpoint_entry['iter_num']:06d}"
                convert(args.python, args.converter, checkpoint, args.tokenizer, output, args.dtype)
                converted_by_path[checkpoint] = output
            revision = label["revision"]
            api.create_branch(args.repo, branch=revision, repo_type="model", exist_ok=True)
            api.upload_folder(
                repo_id=args.repo,
                repo_type="model",
                revision=revision,
                folder_path=output,
                commit_message=(
                    f"{args.series} checkpoint {revision}: iter={checkpoint_entry['iter_num']} "
                    f"words={checkpoint_entry.get('words_seen')} tokens={checkpoint_entry.get('tokens_seen')}"
                ),
            )

        final_output = tmp_root / "final-main"
        convert(args.python, args.converter, final_checkpoint, args.tokenizer, final_output, args.dtype)
        api.upload_folder(
            repo_id=args.repo,
            repo_type="model",
            revision="main",
            folder_path=final_output,
            commit_message=f"final model from {final_checkpoint.name}",
        )

    print(
        f"pushed {len(milestones)} {args.series} revisions + main to {args.repo}; "
        f"main source={final_checkpoint.name}"
    )


if __name__ == "__main__":
    main()
