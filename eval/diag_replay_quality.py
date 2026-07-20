#!/usr/bin/env python3
"""Grade the s1337 baseline replay against the three preserved original checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from diag_parse_results import parse_accuracy_report


LABELS = ("10M", "20M", "30M")
CRITICAL_CONFIG = (
    "dataset", "block_size", "batch_size", "gradient_accumulation_steps", "n_layer",
    "n_head", "n_embd", "dropout", "bias", "use_rmsnorm", "use_swiglu",
    "swiglu_mult", "use_rope", "use_attn_gate", "use_attn_res", "use_muon",
    "use_hybrid", "sampler", "sampler_seed", "seed", "warmup_iters", "lr_decay_iters",
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def plan_index(series_dir: Path) -> dict:
    rows = load_json(series_dir / "plan.json").get("checkpoints")
    if not isinstance(rows, list):
        raise ValueError(f"invalid plan: {series_dir / 'plan.json'}")
    return {row["checkpoint_label"]: row for row in rows}


def report_metrics(eval_root: Path, model_name: str) -> tuple[float, float]:
    path = (
        eval_root / "results" / model_name / "main" / "zero_shot" / "causal"
        / "blimp" / "blimp_filtered" / "best_temperature_report.txt"
    )
    rows = parse_accuracy_report(path)
    overall = next(float(row["accuracy"]) for row in rows if row["level"] == "overall")
    filler = next(
        float(row["accuracy"]) for row in rows
        if row["level"] == "linguistics_term" and row["term"] == "filler_gap_dependency"
    )
    return overall, filler


def tensor_differences(
    reference_path: Path, replay_path: Path
) -> tuple[bool, bool, float, float]:
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    replay = torch.load(replay_path, map_location="cpu", weights_only=False)
    ref_config = reference.get("config", {})
    replay_config = replay.get("config", {})
    config_equal = all(ref_config.get(key) == replay_config.get(key) for key in CRITICAL_CONFIG)
    ref_fingerprints = reference.get("provenance", {}).get("data_fingerprints", {})
    replay_fingerprints = replay.get("provenance", {}).get("data_fingerprints", {})
    if ref_fingerprints != replay_fingerprints:
        config_equal = False
    ref_state = reference["model"]
    replay_state = replay["model"]
    if set(ref_state) != set(replay_state):
        raise RuntimeError("reference and replay tensor keys differ")
    max_abs = 0.0
    abs_sum = 0.0
    count = 0
    exact = True
    for key in sorted(ref_state):
        left = ref_state[key]
        right = replay_state[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"tensor metadata differs for {key}")
        exact = exact and torch.equal(left, right)
        difference = (left.float() - right.float()).abs()
        max_abs = max(max_abs, float(difference.max()))
        abs_sum += float(difference.sum(dtype=torch.float64))
        count += difference.numel()
    del reference, replay, ref_state, replay_state
    return config_equal, exact, max_abs, abs_sum / count


def compare(
    reference_series: Path,
    replay_series: Path,
    eval_root: Path,
    skip_tensors: bool = False,
) -> tuple[list[dict], dict]:
    reference_plan = plan_index(reference_series)
    replay_plan = plan_index(replay_series)
    rows = []
    for label in LABELS:
        reference_item = reference_plan[label]
        replay_item = replay_plan[label]
        reference_dev = load_json(reference_series / label / "dev_loss.json")
        replay_dev = load_json(replay_series / label / "dev_loss.json")
        reference_overall, reference_filler = report_metrics(
            eval_root, reference_item["blimp_model_name"]
        )
        replay_overall, replay_filler = report_metrics(eval_root, replay_item["blimp_model_name"])
        if skip_tensors:
            config_equal, exact, tensor_max, tensor_mean = False, False, math.nan, math.nan
        else:
            config_equal, exact, tensor_max, tensor_mean = tensor_differences(
                Path(reference_item["checkpoint_path"]), Path(replay_item["checkpoint_path"])
            )
        rows.append({
            "checkpoint_label": label,
            "iter_num": replay_item["iter_num"],
            "words_seen": replay_item["words_seen"],
            "config_and_fingerprints_equal": int(config_equal),
            "tensors_exact": int(exact),
            "tensor_max_abs_difference": tensor_max,
            "tensor_mean_abs_difference": tensor_mean,
            "reference_nll": reference_dev["mean_nll"],
            "replay_nll": replay_dev["mean_nll"],
            "absolute_nll_difference": abs(
                float(reference_dev["mean_nll"]) - float(replay_dev["mean_nll"])
            ),
            "reference_overall_blimp": reference_overall,
            "replay_overall_blimp": replay_overall,
            "absolute_overall_blimp_difference_pp": abs(reference_overall - replay_overall),
            "reference_filler_gap": reference_filler,
            "replay_filler_gap": replay_filler,
            "absolute_filler_gap_difference_pp": abs(reference_filler - replay_filler),
        })
    max_nll = max(float(row["absolute_nll_difference"]) for row in rows)
    max_blimp = max(float(row["absolute_overall_blimp_difference_pp"]) for row in rows)
    if max_nll <= 0.01 and max_blimp <= 0.5:
        grade = "green"
    elif max_nll <= 0.02 and max_blimp <= 1.0:
        grade = "yellow"
    else:
        grade = "red"
    summary = {
        "status": "complete",
        "grade": grade,
        "max_nll_difference": max_nll,
        "max_overall_blimp_difference_pp": max_blimp,
        "max_filler_gap_difference_pp": max(
            float(row["absolute_filler_gap_difference_pp"]) for row in rows
        ),
        "config_fingerprint_equal_points": sum(int(row["config_and_fingerprints_equal"]) for row in rows),
        "tensor_equal_points": sum(int(row["tensors_exact"]) for row in rows),
        "tensor_points": len(rows),
        "criterion": "green: NLL<=0.01 and BLiMP<=0.5; yellow: NLL<=0.02 and BLiMP<=1.0",
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-series", type=Path, required=True)
    parser.add_argument("--replay-series", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--skip-tensors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = compare(
        args.reference_series.resolve(),
        args.replay_series.resolve(),
        args.eval_root.resolve(),
        args.skip_tensors,
    )
    if args.output_csv.exists() or args.output_json.exists():
        raise FileExistsError("replay-quality output already exists")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
