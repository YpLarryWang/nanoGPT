#!/usr/bin/env python3
"""Run the official BabyLM AoA computation from local checkpoint directories.

The upstream evaluator assumes one Hugging Face Hub repository whose revisions
are named ``chck_*M``. This wrapper preserves the evaluator and scoring logic,
but maps each manifest milestone to a separately converted local HF directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def local_step_plan(run_dir: Path, manifest: dict, series: str) -> list[dict]:
    milestones = []
    for checkpoint in manifest.get("checkpoints", []):
        if checkpoint.get("role") != "milestone":
            continue
        for label in checkpoint.get("labels", []):
            if label.get("series") == series:
                milestones.append((checkpoint, label))
    milestones.sort(key=lambda pair: pair[0]["iter_num"])

    # A rounded official revision can legitimately name two distinct local
    # points, e.g. exact 90M words and a 90.34M-word final checkpoint both use
    # ``chck_90M``. Hub uploads must still reject that branch collision, but a
    # local AoA curve should retain both distinct x coordinates. Disambiguate
    # only the on-disk conversion cache key.
    revision_counts = Counter(label["revision"] for _, label in milestones)
    plan = []
    seen_steps = set()
    for checkpoint, label in milestones:
        step = int(label["target"])
        if step in seen_steps:
            raise ValueError(f"duplicate {series} target count: {step}")
        seen_steps.add(step)
        source = run_dir / checkpoint["path"]
        revision = label["revision"]
        cache_revision = revision
        if revision_counts[revision] > 1:
            cache_revision = (
                f"{revision}-t{step}-i{int(checkpoint['iter_num']):06d}"
            )
        plan.append(
            {
                "step": step,
                "revision": revision,
                "cache_revision": cache_revision,
                "source": source,
                "iter_num": int(checkpoint["iter_num"]),
                "actual": int(label.get("actual", step)),
                "sha256": checkpoint.get("sha256"),
            }
        )
    return plan


def _finite_surprisal(result: dict) -> bool:
    value = result.get("surprisal")
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def split_complete_resume_results(
    plan: list[dict], existing_results: dict | None, expected_per_step: int
) -> tuple[list[dict], set[int], dict[int, dict[str, int]]]:
    """Keep only complete, finite checkpoint results from a resume file."""
    if expected_per_step <= 0:
        raise ValueError("expected_per_step must be positive")

    plan_steps = {int(item["step"]) for item in plan}
    grouped: dict[int, list[dict]] = {step: [] for step in plan_steps}
    if existing_results:
        for result in existing_results.get("results", []):
            if "step" not in result:
                continue
            step = int(result["step"])
            if step in grouped:
                grouped[step].append(result)

    kept = []
    completed = set()
    rejected = {}
    for item in plan:
        step = int(item["step"])
        rows = grouped[step]
        finite = sum(_finite_surprisal(row) for row in rows)
        identities = {
            (row.get("target_word"), row.get("context_id"), row.get("context"))
            for row in rows
        }
        if (
            len(rows) == expected_per_step
            and finite == expected_per_step
            and len(identities) == expected_per_step
        ):
            kept.extend(rows)
            completed.add(step)
        elif rows:
            rejected[step] = {
                "rows": len(rows),
                "finite": finite,
                "unique": len(identities),
                "expected": expected_per_step,
            }
    return kept, completed, rejected


def unfinished_steps(plan: list[dict], completed_steps: set[int]) -> list[dict]:
    return [item for item in plan if int(item["step"]) not in completed_steps]


def validate_complete_results(
    plan: list[dict], results: dict, expected_per_step: int
) -> None:
    kept, completed, rejected = split_complete_resume_results(
        plan, results, expected_per_step
    )
    expected_steps = {int(item["step"]) for item in plan}
    if rejected or completed != expected_steps:
        missing = sorted(expected_steps - completed)
        raise RuntimeError(
            "incomplete AoA results: "
            f"missing_steps={missing} rejected_steps={rejected}"
        )
    expected_total = expected_per_step * len(plan)
    if len(kept) != expected_total:
        raise RuntimeError(
            f"AoA result count mismatch: {len(kept)} != {expected_total}"
        )


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

    output = cache_dir / item["cache_revision"]
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

    local_paths = {
        item["step"]: cache_dir / item["cache_revision"] for item in full_plan
    }

    sys.path.insert(0, str(eval_root))
    os.chdir(eval_root)
    import torch
    from tqdm import tqdm
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

        def analyze_steps(
            self,
            contexts,
            target_words,
            use_bos_only=False,
            resume_path=None,
        ):
            """Official extraction loop with CUDA/non-finite failures made fatal."""
            existing = []
            if resume_path and resume_path.is_file():
                existing_data = JsonProcessor.load_json(resume_path)
                existing = existing_data.get("results", [])

            new_results = []
            for step, word_count in zip(
                self.config.steps, self.config.word_counts, strict=True
            ):
                print(f"Checkpoint: {step}")
                model = processor = tokenizer = None
                try:
                    model = self.load_model_for_step(step)
                    processor, tokenizer = self.load_tokenizer_for_step(step)
                    for word_contexts, target_word in tqdm(
                        zip(contexts, target_words, strict=False),
                        total=len(target_words),
                    ):
                        for context_idx, context in enumerate(word_contexts):
                            surprisal = self.compute_surprisal(
                                model,
                                processor,
                                tokenizer,
                                context,
                                target_word,
                                use_bos_only=use_bos_only,
                            )
                            if not math.isfinite(float(surprisal)):
                                raise RuntimeError(
                                    "non-finite surprisal: "
                                    f"step={step} word={target_word!r} "
                                    f"context_id={context_idx}"
                                )
                            new_results.append(
                                {
                                    "step": step,
                                    "word_count": word_count,
                                    "target_word": target_word,
                                    "context_id": context_idx,
                                    "context": "BOS_ONLY" if use_bos_only else context,
                                    "surprisal": surprisal,
                                }
                            )

                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    if resume_path:
                        combined = existing + new_results
                        JsonProcessor.save_json(
                            {
                                "metadata": {
                                    "model_name": self.model_name,
                                    "use_bos_only": use_bos_only,
                                    "total_steps": len(self.config.steps),
                                    "completed_steps": len(
                                        {int(row["step"]) for row in combined}
                                    ),
                                },
                                "results": combined,
                            },
                            resume_path,
                        )
                finally:
                    del model, processor, tokenizer
                    if self.device == "cuda":
                        torch.cuda.empty_cache()

            combined = existing + new_results
            return {
                "metadata": {
                    "model_name": self.model_name,
                    "use_bos_only": use_bos_only,
                    "total_steps": len(self.config.steps),
                    "completed_steps": len(
                        {int(row["step"]) for row in combined}
                    ),
                },
                "results": combined,
            }

    target_words, contexts = load_eval(word_path, args.min_context, debug=False)
    if args.max_words is not None:
        target_words = target_words[: args.max_words]
        contexts = contexts[: args.max_words]
    if args.max_contexts is not None:
        contexts = [items[: args.max_contexts] for items in contexts]

    expected_per_step = sum(map(len, contexts))
    result_dir = output_dir / run_dir.name
    result_dir.mkdir(parents=True, exist_ok=True)
    resume_file = result_dir / "resume" / "surprisal.json" if args.resume else None
    existing_results = None
    if resume_file and resume_file.is_file():
        existing_results = JsonProcessor.load_json(resume_file)
    kept_results, completed_steps, rejected_steps = split_complete_resume_results(
        full_plan, existing_results, expected_per_step
    )
    if rejected_steps:
        print(f"discarding incomplete resume steps: {rejected_steps}")
    if resume_file:
        resume_file.parent.mkdir(parents=True, exist_ok=True)
        existing_results = {
            "metadata": {
                "model_name": str(cache_dir),
                "total_steps": len(full_plan),
                "completed_steps": len(completed_steps),
                "sanitized": True,
            },
            "results": kept_results,
        }
        JsonProcessor.save_json(existing_results, resume_file)
    plan = unfinished_steps(full_plan, completed_steps)

    for item in plan:
        local_paths[item["step"]] = ensure_converted(
            item,
            cache_dir=cache_dir,
            converter=converter,
            tokenizer=tokenizer_path,
            python=args.python,
            dtype=args.dtype,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if plan:
        class LocalConfig:
            steps = [item["step"] for item in plan]
            word_counts = [item["step"] for item in plan]

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
    elif existing_results and kept_results:
        results = existing_results
    else:
        raise RuntimeError("no unfinished checkpoints and no resumable results")

    validate_complete_results(full_plan, results, expected_per_step)
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
    curve_fitness = float(score.get("curve_fitness", float("nan")))
    if not math.isfinite(curve_fitness):
        raise RuntimeError(f"non-finite AoA curve_fitness: {curve_fitness}")
    JsonProcessor.save_json({"aoa": score}, result_dir / "aoa_score.json")
    print(
        f"local AoA complete: checkpoints={len(full_plan)} words={len(target_words)} "
        f"contexts={sum(map(len, contexts))} device={device} output={result_dir}"
    )


if __name__ == "__main__":
    main()
