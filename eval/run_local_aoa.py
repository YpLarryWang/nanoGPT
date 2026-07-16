#!/usr/bin/env python3
"""Run the official BabyLM AoA computation from local checkpoint directories.

The upstream evaluator assumes one Hugging Face Hub repository whose revisions
are named ``chck_*M``. This wrapper preserves the evaluator and scoring logic,
but maps each manifest milestone to a separately converted local HF directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from .push_checkpoint_series import selected_milestones
except ImportError:
    from push_checkpoint_series import selected_milestones


def local_step_plan(run_dir: Path, manifest: dict, series: str) -> list[dict]:
    plan = []
    seen_steps = set()
    for checkpoint, label in selected_milestones(manifest, series):
        step = int(label["target"])
        if step in seen_steps:
            raise ValueError(f"duplicate {series} target count: {step}")
        seen_steps.add(step)
        source = run_dir / checkpoint["path"]
        plan.append(
            {
                "step": step,
                "revision": label["revision"],
                "source": source,
                "iter_num": int(checkpoint["iter_num"]),
                "actual": int(label.get("actual", step)),
                "sha256": checkpoint.get("sha256"),
            }
        )
    return plan


def unfinished_steps(plan: list[dict], existing_results: dict | None) -> list[dict]:
    if not existing_results:
        return plan
    completed = {
        int(result["step"])
        for result in existing_results.get("results", [])
        if "step" in result
    }
    return [item for item in plan if item["step"] not in completed]


def ensure_converted(
    item: dict,
    *,
    cache_dir: Path,
    converter: Path,
    tokenizer: Path,
    python: str,
    dtype: str,
) -> Path:
    source = item["source"]
    if not source.is_file():
        raise FileNotFoundError(f"missing checkpoint: {source}")

    output = cache_dir / item["revision"]
    metadata_path = output / "checkpoint_source.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_sha = item.get("sha256")
        if expected_sha and metadata.get("sha256") != expected_sha:
            raise RuntimeError(
                f"cached conversion SHA mismatch for {item['revision']}: "
                f"{metadata.get('sha256')} != {expected_sha}"
            )
        return output
    if output.exists():
        raise RuntimeError(f"incomplete cached conversion exists: {output}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            python,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--series", default="words", choices=["words", "tokens"])
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path(__file__).with_name("convert_nanogpt_to_hf.py"),
    )
    parser.add_argument("--word-path", type=Path)
    parser.add_argument("--min-context", type=int, default=0)
    parser.add_argument("--max-checkpoints", type=int)
    parser.add_argument("--max-words", type=int)
    parser.add_argument("--max-contexts", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    eval_root = args.eval_root.resolve()
    tokenizer_path = args.tokenizer.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    converter = args.converter.resolve()
    word_path = (
        args.word_path.resolve()
        if args.word_path
        else eval_root / "evaluation_data/full_eval/aoa/cdi_childes.json"
    )

    manifest = json.loads((run_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    full_plan = local_step_plan(run_dir, manifest, args.series)
    if args.max_checkpoints is not None:
        full_plan = full_plan[: args.max_checkpoints]
    if not full_plan:
        raise SystemExit(f"manifest has no {args.series!r} checkpoints")

    result_dir = output_dir / run_dir.name
    result_dir.mkdir(parents=True, exist_ok=True)
    resume_file = result_dir / "resume" / "surprisal.json" if args.resume else None
    existing_results = None
    if resume_file and resume_file.is_file():
        existing_results = json.loads(resume_file.read_text(encoding="utf-8"))
    plan = unfinished_steps(full_plan, existing_results)

    local_paths = {item["step"]: cache_dir / item["revision"] for item in full_plan}
    for item in plan:
        local_paths[item["step"]] = ensure_converted(
            item,
            cache_dir=cache_dir,
            converter=converter,
            tokenizer=tokenizer_path,
            python=args.python,
            dtype=args.dtype,
        )

    sys.path.insert(0, str(eval_root))
    os.chdir(eval_root)
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    from evaluation_pipeline.AoA_word.eval_util import JsonProcessor, load_eval
    from evaluation_pipeline.AoA_word.evaluation_functions import StepSurprisalExtractor
    from evaluation_pipeline.utils import AoAEvaluator

    class LocalStepSurprisalExtractor(StepSurprisalExtractor):
        def _path(self, step: int) -> Path:
            return local_paths[int(step)]

        def load_model_for_step(self, step: int):
            model = AutoModelForCausalLM.from_pretrained(
                self._path(step), trust_remote_code=True
            )
            return model.to(self.device).eval()

        def load_tokenizer_for_step(self, step: int):
            processor = AutoProcessor.from_pretrained(
                self._path(step), trust_remote_code=True, padding_side="right"
            )
            tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            return processor, tokenizer

    target_words, contexts = load_eval(word_path, args.min_context, debug=False)
    if args.max_words is not None:
        target_words = target_words[: args.max_words]
        contexts = contexts[: args.max_words]
    if args.max_contexts is not None:
        contexts = [items[: args.max_contexts] for items in contexts]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if plan:
        class LocalConfig:
            steps = [item["step"] for item in plan]
            word_counts = [item["step"] for item in plan]

        if resume_file:
            resume_file.parent.mkdir(parents=True, exist_ok=True)
        extractor = LocalStepSurprisalExtractor(
            config=LocalConfig(),
            model_name=str(cache_dir),
            backend="causal",
            device=device,
        )
        results = extractor.analyze_steps(
            contexts=contexts,
            target_words=target_words,
            resume_path=resume_file,
        )
    elif existing_results:
        results = existing_results
    else:
        raise RuntimeError("no unfinished checkpoints and no resumable results")

    results["metadata"].update(
        {
            "total_steps": len(full_plan),
            "completed_steps": len({result["step"] for result in results["results"]}),
            "run_name": manifest.get("run_name", run_dir.name),
            "series": args.series,
            "min_context": args.min_context,
            "local_checkpoint_paths": True,
            "checkpoint_plan": [
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in item.items()
                    if key != "sha256"
                }
                for item in full_plan
            ],
        }
    )

    result_file = result_dir / "surprisal.json"
    JsonProcessor.save_json(results, result_file)

    score = AoAEvaluator(word_path.parent / "cdi_human.csv").compute_curve_fitness(
        results,
        AutoTokenizer.from_pretrained(
            local_paths[full_plan[-1]["step"]], trust_remote_code=True
        ),
    )
    JsonProcessor.save_json({"aoa": score}, result_dir / "aoa_score.json")
    print(
        f"local AoA complete: checkpoints={len(full_plan)} words={len(target_words)} "
        f"contexts={sum(map(len, contexts))} device={device} output={result_dir}"
    )


if __name__ == "__main__":
    main()
