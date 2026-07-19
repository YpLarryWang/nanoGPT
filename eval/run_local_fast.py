#!/usr/bin/env python3
"""Run official BabyLM fast eval from local checkpoint manifests.

The upstream runner expects Hugging Face Hub revisions. This wrapper converts
one local token-series checkpoint at a time into a temporary HF directory,
uses the revision name only for the official results layout, and deletes the
temporary conversion after the five core fast tasks and two GlobalPIQA tasks
finish.
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
STRICT_REVISIONS = {
    *STRICT_SMALL_REVISIONS,
    *(f"chck_{i}M" for i in range(200, 1001, 100)),
}
TRACK_REVISIONS = {
    "strict-small": STRICT_SMALL_REVISIONS,
    "strict": STRICT_REVISIONS,
}


def checkpoint_plan(run_dir: Path, manifest: dict, track: str) -> list[dict]:
    required_revisions = TRACK_REVISIONS[track]
    plan = []
    for checkpoint, label in selected_milestones(manifest, "tokens"):
        revision = label["revision"]
        if revision not in required_revisions:
            continue
        plan.append(
            {
                "revision": revision,
                "source": run_dir / checkpoint["path"],
                "iter_num": int(checkpoint["iter_num"]),
                "tokens_seen": int(checkpoint["tokens_seen"]),
            }
        )
    missing = required_revisions - {item["revision"] for item in plan}
    if missing:
        raise ValueError(f"manifest is missing {track} revisions: {sorted(missing)}")
    return plan


def prediction_files(eval_root: Path, model_name: str, revision: str) -> list[Path]:
    revision_dir = eval_root / "results" / model_name / revision
    return sorted(revision_dir.rglob("predictions.json")) if revision_dir.is_dir() else []


def required_prediction_files(
    eval_root: Path, model_name: str, revision: str
) -> tuple[list[Path], list[Path]]:
    root = eval_root / "results" / model_name / revision / "zero_shot" / "causal"
    core = [
        root / "blimp" / "blimp_fast" / "predictions.json",
        root / "blimp" / "supplement_fast" / "predictions.json",
        root / "ewok" / "ewok_fast" / "predictions.json",
        root / "entity_tracking" / "entity_tracking_fast" / "predictions.json",
        root / "reading" / "predictions.json",
    ]
    global_piqa = [
        root / task / task / "predictions.json"
        for task in ("global_piqa_parallel", "global_piqa_nonparallel")
    ]
    return core, global_piqa


def run_global_piqa(
    *, python: str, eval_root: Path, model: Path, revision: str, env: dict[str, str]
) -> None:
    for task in ("global_piqa_parallel", "global_piqa_nonparallel"):
        subprocess.run(
            [
                python,
                "-m",
                "evaluation_pipeline.sentence_zero_shot.run",
                "--model_path_or_name",
                str(model),
                "--backend",
                "causal",
                "--task",
                task,
                "--data_path",
                f"evaluation_data/fast_eval/{task}",
                "--save_predictions",
                "--revision_name",
                revision,
            ],
            cwd=eval_root,
            env=env,
            check=True,
        )


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
    parser.add_argument("--track", required=True, choices=sorted(TRACK_REVISIONS))
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
    plan = checkpoint_plan(run_dir, manifest, args.track)
    if args.revisions:
        wanted = set(args.revisions)
        plan = [item for item in plan if item["revision"] in wanted]
        unknown = wanted - {item["revision"] for item in plan}
        if unknown:
            raise SystemExit(
                f"requested revision not found in {args.track} plan: {sorted(unknown)}"
            )
    if args.max_checkpoints is not None:
        plan = plan[: args.max_checkpoints]

    missing_sources = [item["source"] for item in plan if not item["source"].is_file()]
    if missing_sources:
        raise SystemExit(f"missing checkpoint: {missing_sources[0]}")
    scratch_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # Keep the venv's bin directory itself on PATH. Resolving the ``python``
    # symlink would point at /usr/bin and make the upstream shell script lose
    # the venv's bare ``python`` executable.
    env["PATH"] = f"{Path(args.python).expanduser().absolute().parent}:{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    completed = 0
    for index, item in enumerate(plan, start=1):
        revision = item["revision"]
        core_files, global_files = required_prediction_files(eval_root, model_name, revision)
        missing_core = [path for path in core_files if not path.is_file()]
        missing_global = [path for path in global_files if not path.is_file()]
        if args.resume and not missing_core and not missing_global:
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
            if not args.resume or missing_core:
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
            if not args.resume or missing_global:
                run_global_piqa(
                    python=args.python,
                    eval_root=eval_root,
                    model=local_model,
                    revision=revision,
                    env=env,
                )

        required = [*core_files, *global_files]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"{revision} is missing {len(missing)}/7 prediction files: "
                f"{[str(path) for path in missing]}"
            )
        size = sum(path.stat().st_size for path in required)
        print(f"[{index}/{len(plan)}] {revision}: complete; predictions={size} bytes", flush=True)
        completed += 1

    print(
        f"local fast eval complete: model={model_name} checkpoints={completed}/{len(plan)} "
        f"results={eval_root / 'results' / model_name}",
        flush=True,
    )


if __name__ == "__main__":
    main()
