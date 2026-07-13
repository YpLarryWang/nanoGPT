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


def role_checkpoint(run_dir: Path, manifest: dict, role: str) -> Path:
    relative_path = manifest.get("roles", {}).get(role)
    if not relative_path:
        raise ValueError(f"manifest has no {role!r} checkpoint")
    checkpoint = run_dir / relative_path
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing {role} checkpoint: {checkpoint}")
    return checkpoint


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
    parser.add_argument("--main-role", default="final", choices=["final", "best"])
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    with (run_dir / "checkpoint_manifest.json").open(encoding="utf-8") as f:
        manifest = json.load(f)
    milestones = selected_milestones(manifest, args.series)
    if not milestones:
        raise SystemExit(f"manifest contains no {args.series!r} milestone checkpoints")

    try:
        main_checkpoint = role_checkpoint(run_dir, manifest, args.main_role)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc

    missing = [
        run_dir / entry["path"]
        for entry, _ in milestones
        if not (run_dir / entry["path"]).is_file()
    ]
    if missing:
        raise SystemExit(f"missing milestone checkpoint: {missing[0]}")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="babylm-hf-") as tmp:
        tmp_root = Path(tmp)

        # A newly-created Hub repo has no commit from which a revision branch can
        # be created. Seed main first, then fork and populate the milestone branches.
        main_output = tmp_root / f"{args.main_role}-main"
        convert(args.python, args.converter, main_checkpoint, args.tokenizer, main_output, args.dtype)
        api.upload_folder(
            repo_id=args.repo,
            repo_type="model",
            revision="main",
            folder_path=main_output,
            commit_message=f"{args.main_role} model from {main_checkpoint.name}",
        )

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

    print(
        f"pushed {len(milestones)} {args.series} revisions + main to {args.repo}; "
        f"main role={args.main_role} source={main_checkpoint.name}"
    )


if __name__ == "__main__":
    main()
