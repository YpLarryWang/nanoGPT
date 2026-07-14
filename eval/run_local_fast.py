#!/usr/bin/env python3
"""Run official BabyLM strict-small fast eval from local checkpoint manifests.

The upstream runner expects Hugging Face Hub revisions. This wrapper converts
one local token-series checkpoint at a time into a temporary HF directory,
uses the revision name only for the official results layout, and deletes the
temporary conversion after its five fast tasks finish.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from .push_checkpoint_series import selected_milestones
except ImportError:
    from push_checkpoint_series import selected_milestones


STRICT_SMALL_REVISIONS = {
    *(f"chck_{i}M" for i in range(1, 10)),
    *(f"chck_{i}M" for i in range(10, 101, 10)),
}


def strict_small_plan(run_dir: Path, manifest: dict) -> list[dict]:
    plan = []
    for checkpoint, label in selected_milestones(manifest, "tokens"):
        revision = label["revision"]
        if revision not in STRICT_SMALL_REVISIONS:
            continue
        plan.append(
            {
                "revision": revision,
                "source": run_dir / checkpoint["path"],
                "iter_num": int(checkpoint["iter_num"]),
                "tokens_seen": int(checkpoint["tokens_seen"]),
            }
        )
    missing = STRICT_SMALL_REVISIONS - {item["revision"] for item in plan}
    if missing:
        raise ValueError(f"manifest is missing strict-small revisions: {sorted(missing)}")
    return plan


def prediction_files(eval_root: Path, model_name: str, revision: str) -> list[Path]:
    revision_dir = eval_root / "results" / model_name / revision
    return sorted(revision_dir.rglob("predictions.json")) if revision_dir.is_dir() else []


def convert_checkpoint(
    *, python: str, converter: Path, checkpoint: Path, tokenizer: Path, output: Path, dtype: str
) -> None:
    subprocess.run(
        [
            python,
            str(converter),
            "--ckpt",
            str(checkpoint),
            "--tokenizer",
            str(tokenizer),
            "--out",
            str(output),
            "--dtype",
            dtype,
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path(__file__).with_name("convert_nanogpt_to_hf.py"),
    )
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--revision", action="append", dest="revisions")
    parser.add_argument("--max-checkpoints", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    eval_root = args.eval_root.resolve()
    tokenizer = args.tokenizer.resolve()
    converter = args.converter.resolve()
    scratch_root = args.scratch_root.resolve()
    manifest = json.loads((run_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    model_name = args.model_name or manifest.get("run_name") or run_dir.name
    plan = strict_small_plan(run_dir, manifest)
    if args.revisions:
        wanted = set(args.revisions)
        plan = [item for item in plan if item["revision"] in wanted]
        unknown = wanted - {item["revision"] for item in plan}
        if unknown:
            raise SystemExit(f"requested revision not found in strict-small plan: {sorted(unknown)}")
    if args.max_checkpoints is not None:
        plan = plan[: args.max_checkpoints]

    missing_sources = [item["source"] for item in plan if not item["source"].is_file()]
    if missing_sources:
        raise SystemExit(f"missing checkpoint: {missing_sources[0]}")
    scratch_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PATH"] = f"{Path(args.python).resolve().parent}:{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    completed = 0
    for index, item in enumerate(plan, start=1):
        revision = item["revision"]
        existing = prediction_files(eval_root, model_name, revision)
        if args.resume and len(existing) == 5:
            print(f"[{index}/{len(plan)}] {revision}: already complete; skipping", flush=True)
            completed += 1
            continue

        print(
            f"[{index}/{len(plan)}] {revision}: converting {item['source'].name} "
            f"(iter={item['iter_num']} tokens={item['tokens_seen']})",
            flush=True,
        )
        with tempfile.TemporaryDirectory(prefix=f"{model_name}-{revision}-", dir=scratch_root) as tmp:
            local_model = Path(tmp) / model_name
            convert_checkpoint(
                python=args.python,
                converter=converter,
                checkpoint=item["source"],
                tokenizer=tokenizer,
                output=local_model,
                dtype=args.dtype,
            )
            subprocess.run(
                [
                    "bash",
                    "scripts/eval_zero_shot_fast.sh",
                    str(local_model),
                    revision,
                    "causal",
                ],
                cwd=eval_root,
                env=env,
                check=True,
            )

        produced = prediction_files(eval_root, model_name, revision)
        if len(produced) != 5:
            raise RuntimeError(
                f"{revision} produced {len(produced)}/5 prediction files: "
                f"{[str(path) for path in produced]}"
            )
        size = sum(path.stat().st_size for path in produced)
        print(f"[{index}/{len(plan)}] {revision}: complete; predictions={size} bytes", flush=True)
        completed += 1

    print(
        f"local fast eval complete: model={model_name} checkpoints={completed}/{len(plan)} "
        f"results={eval_root / 'results' / model_name}",
        flush=True,
    )


if __name__ == "__main__":
    main()
