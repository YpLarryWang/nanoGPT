#!/usr/bin/env python3
"""Summarize BabyLM GLUE fine-tuning using accuracy for all seven tasks.

This intentionally does not update the legacy GLUE scoreboards: their
``macro6``/``macro7`` columns use the historical task-specific metric mix.
The all-accuracy mean emitted here is the value destined for
``offdev_results.csv:superglue_ld``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = ("boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli")


def read_accuracy(path: Path) -> float:
    metrics: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            metrics[key.strip()] = float(value.strip())
        except ValueError:
            continue
    if "accuracy" not in metrics:
        raise ValueError(f"missing accuracy in {path}")
    accuracy = metrics["accuracy"]
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"accuracy outside [0, 1] in {path}: {accuracy}")
    return accuracy * 100.0


def collect(eval_repo: Path, model: str, revision: str = "main") -> dict[str, object]:
    base = eval_repo / "results" / model / revision / "finetune"
    scores = {task: read_accuracy(base / task / "results.txt") for task in TASKS}
    return {
        "model": model,
        "revision": revision,
        "metric_protocol": "all_accuracy",
        "task_accuracy": {task: round(scores[task], 2) for task in TASKS},
        "superglue_ld": round(sum(scores.values()) / len(scores), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--eval-repo", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--write-json", type=Path)
    args = parser.parse_args()

    summary = collect(args.eval_repo, args.model, args.revision)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(rendered)
        print(f"wrote: {args.write_json}")


if __name__ == "__main__":
    main()
