#!/usr/bin/env python3
"""Run the frozen six-point L32 diagnosis grid without retraining.

The driver itself uses only the standard library. It invokes native
``full_dev_loss.py`` with the training environment and BLiMP conversion/scoring
with the separately pinned BabyLM evaluation environment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


CHECKPOINT_LABELS = ("1M", "5M", "10M", "20M", "50M", "final")
MILESTONE_NAMES = {label: f"words_{label}" for label in CHECKPOINT_LABELS[:-1]}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def diagnosis_plan(run_dir: Path, manifest: dict) -> list[dict]:
    """Select the five named word milestones plus the physical final role."""
    checkpoints = manifest.get("checkpoints", [])
    selected: dict[str, dict] = {}
    for entry in checkpoints:
        if entry.get("role") != "milestone":
            continue
        names = {
            label.get("name")
            for label in entry.get("labels", [])
            if label.get("series") == "words"
        }
        for short_label, manifest_name in MILESTONE_NAMES.items():
            if manifest_name not in names:
                continue
            if short_label in selected:
                raise ValueError(f"duplicate milestone label {manifest_name!r}")
            selected[short_label] = entry

    final_path = manifest.get("roles", {}).get("final")
    if not final_path:
        raise ValueError("manifest is missing roles.final")
    final_entries = [
        entry
        for entry in checkpoints
        if entry.get("role") == "final" and entry.get("path") == final_path
    ]
    if len(final_entries) != 1:
        raise ValueError(
            f"expected one physical final entry for {final_path!r}, got {len(final_entries)}"
        )
    selected["final"] = final_entries[0]

    missing = [label for label in CHECKPOINT_LABELS if label not in selected]
    if missing:
        raise ValueError(f"manifest is missing diagnosis checkpoints: {missing}")

    run_name = manifest.get("run_name") or run_dir.name
    plan = []
    for label in CHECKPOINT_LABELS:
        entry = selected[label]
        source = run_dir / entry["path"]
        plan.append(
            {
                "run_name": run_name,
                "checkpoint_label": label,
                "checkpoint_path": str(source),
                "checkpoint_filename": entry["path"],
                "checkpoint_role": entry["role"],
                "iter_num": int(entry["iter_num"]),
                "tokens_seen": int(entry["tokens_seen"]),
                "words_seen": int(entry["words_seen"]),
                "sha256": entry.get("sha256"),
                "blimp_model_name": (
                    run_name if label == "final" else f"{run_name}--diag-words-{label}"
                ),
            }
        )
    return plan


def blimp_paths(eval_root: Path, model_name: str) -> tuple[Path, Path]:
    root = (
        eval_root
        / "results"
        / model_name
        / "main"
        / "zero_shot"
        / "causal"
        / "blimp"
        / "blimp_filtered"
    )
    return root / "best_temperature_report.txt", root / "predictions.json"


def blimp_complete(eval_root: Path, model_name: str) -> bool:
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in blimp_paths(eval_root, model_name)
    )


def ensure_converted(
    item: dict,
    *,
    cache_root: Path,
    converter: Path,
    tokenizer: Path,
    eval_python: str,
    dtype: str,
) -> Path:
    source = Path(item["checkpoint_path"])
    if not source.is_file():
        raise FileNotFoundError(source)
    output = cache_root / item["blimp_model_name"]
    metadata_path = output / "checkpoint_source.json"
    if metadata_path.is_file():
        metadata = load_json(metadata_path)
        if metadata.get("filename") != source.name:
            raise RuntimeError(
                f"cached conversion source mismatch: {metadata.get('filename')} != {source.name}"
            )
        if int(metadata.get("iter_num")) != item["iter_num"]:
            raise RuntimeError("cached conversion iter_num mismatch")
        expected_sha = item.get("sha256")
        if expected_sha and metadata.get("sha256") != expected_sha:
            raise RuntimeError("cached conversion checkpoint SHA mismatch")
        cached_modeling = output / "modeling_nanogpt.py"
        source_modeling = converter.parent / "hf_nanogpt" / "modeling_nanogpt.py"
        if (
            not cached_modeling.is_file()
            or cached_modeling.read_bytes() != source_modeling.read_bytes()
        ):
            raise RuntimeError(f"cached conversion has stale modeling code: {output}")
        return output
    if output.exists():
        raise RuntimeError(f"incomplete cached conversion exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            eval_python,
            str(converter),
            "--ckpt",
            str(source),
            "--tokenizer",
            str(tokenizer),
            "--out",
            str(output),
            "--dtype",
            dtype,
        ],
        check=True,
    )
    return output


def write_plan(path: Path, plan: list[dict], resume: bool) -> None:
    payload = {"schema_version": 1, "checkpoints": plan}
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        if load_json(path) != payload:
            raise RuntimeError(f"existing plan differs from current manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_dev_loss(args, item: dict, run_output: Path) -> None:
    checkpoint_output = run_output / item["checkpoint_label"]
    json_path = checkpoint_output / "dev_loss.json"
    position_path = checkpoint_output / "position_nll.csv"
    if json_path.is_file() and position_path.is_file():
        if args.resume:
            existing = load_json(json_path)
            if (
                int(existing.get("iter_num")) == item["iter_num"]
                and existing.get("checkpoint_role") == item["checkpoint_role"]
            ):
                print(f"dev {item['checkpoint_label']}: complete; skipping", flush=True)
                return
            raise RuntimeError(f"existing dev metadata mismatch: {json_path}")
        raise FileExistsError(json_path)
    if json_path.exists() or position_path.exists():
        raise RuntimeError(f"incomplete dev output exists: {checkpoint_output}")
    command = [
        args.native_python,
        str(args.full_dev_loss),
        "--checkpoint",
        item["checkpoint_path"],
        "--data-dir",
        str(args.data_dir),
        "--split",
        "dev",
        "--device",
        args.device,
        "--dtype",
        args.dev_dtype,
        "--output-json",
        str(json_path),
        "--position-csv",
        str(position_path),
    ]
    if item["checkpoint_role"] == "milestone":
        command.append("--allow-milestone")
    if args.batch_size is not None:
        command.extend(("--batch-size", str(args.batch_size)))
    subprocess.run(command, check=True)


def run_blimp(args, item: dict) -> None:
    model_name = item["blimp_model_name"]
    if blimp_complete(args.eval_root, model_name):
        if item["checkpoint_label"] == "final" or args.resume:
            print(f"BLiMP {item['checkpoint_label']}: complete; reusing {model_name}", flush=True)
            return
        raise FileExistsError(blimp_paths(args.eval_root, model_name)[0])
    result_root = args.eval_root / "results" / model_name
    if result_root.exists():
        raise RuntimeError(f"incomplete BLiMP result tree exists: {result_root}")
    local_model = ensure_converted(
        item,
        cache_root=args.cache_root,
        converter=args.converter,
        tokenizer=args.tokenizer,
        eval_python=args.eval_python,
        dtype=args.hf_dtype,
    )
    env = os.environ.copy()
    env["PATH"] = f"{Path(args.eval_python).expanduser().absolute().parent}:{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(
        [
            args.eval_python,
            "-m",
            "evaluation_pipeline.sentence_zero_shot.run",
            "--model_path_or_name",
            str(local_model),
            "--backend",
            "causal",
            "--task",
            "blimp",
            "--data_path",
            "evaluation_data/full_eval/blimp_filtered",
            "--save_predictions",
            "--revision_name",
            "main",
        ],
        cwd=args.eval_root,
        env=env,
        check=True,
    )
    if not blimp_complete(args.eval_root, model_name):
        raise RuntimeError(f"BLiMP did not produce both report and predictions: {model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--eval-root", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--native-python", default=sys.executable)
    parser.add_argument("--eval-python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dev-dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    parser.add_argument(
        "--hf-dtype",
        default="float32",
        choices=("float32", "float16", "bfloat16"),
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--label", action="append", choices=CHECKPOINT_LABELS, dest="labels")
    parser.add_argument("--skip-dev", action="store_true")
    parser.add_argument("--skip-blimp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--full-dev-loss",
        type=Path,
        default=Path(__file__).with_name("full_dev_loss.py"),
    )
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path(__file__).with_name("convert_nanogpt_to_hf.py"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.full_dev_loss = args.full_dev_loss.resolve()
    args.converter = args.converter.resolve()
    manifest = load_json(args.run_dir / "checkpoint_manifest.json")
    full_plan = diagnosis_plan(args.run_dir, manifest)
    plan = full_plan
    if args.labels:
        wanted = set(args.labels)
        plan = [item for item in plan if item["checkpoint_label"] in wanted]
    run_name = manifest.get("run_name") or args.run_dir.name
    for item in plan:
        source = Path(item["checkpoint_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
    print(json.dumps({"run_name": run_name, "checkpoints": plan}, indent=2), flush=True)
    if args.plan_only:
        return
    required = {
        "--data-dir": args.data_dir,
        "--eval-root": args.eval_root,
        "--tokenizer": args.tokenizer,
        "--output-root": args.output_root,
        "--cache-root": args.cache_root,
        "--eval-python": args.eval_python,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"non-plan execution requires: {' '.join(missing)}")
    args.data_dir = args.data_dir.resolve()
    args.eval_root = args.eval_root.resolve()
    args.tokenizer = args.tokenizer.resolve()
    args.output_root = args.output_root.resolve()
    args.cache_root = args.cache_root.resolve()
    run_output = args.output_root / run_name
    write_plan(run_output / "plan.json", full_plan, args.resume)
    for index, item in enumerate(plan, start=1):
        print(
            f"[{index}/{len(plan)}] {item['checkpoint_label']} "
            f"role={item['checkpoint_role']} iter={item['iter_num']} "
            f"words={item['words_seen']}",
            flush=True,
        )
        if not args.skip_dev:
            run_dev_loss(args, item, run_output)
        if not args.skip_blimp:
            run_blimp(args, item)


if __name__ == "__main__":
    main()
