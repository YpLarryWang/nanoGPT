#!/usr/bin/env python3
"""Collate synced diagnosis artifacts into four validated long-form CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


TASK_LEAVES = {
    "blimp": "blimp_filtered",
    "comps": "comps",
    "entity_tracking": "entity_tracking",
}
EXPECTED_BLIMP_TERMS = 13
DEFAULT_MASK_MODES = ("old", "embed", "random_count_matched")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def run_metadata(run_name: str) -> tuple[str, int]:
    architecture = "attnres" if "attnres" in run_name else "baseline"
    match = re.search(r"-s(\d+)(?:$|--)", run_name)
    seed = int(match.group(1)) if match else 1337
    return architecture, seed


def parse_accuracy_report(path: Path) -> list[dict]:
    """Parse all named accuracy sections without depending on task internals."""
    rows = []
    section = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("TEMPERATURE:"):
            continue
        if line.startswith("### "):
            heading = line[4:].strip()
            section = heading.removesuffix(" ACCURACY").lower()
            continue
        if section is None:
            raise ValueError(f"value outside a report section in {path}: {line!r}")
        if section == "average":
            try:
                accuracy = float(line)
            except ValueError as error:
                raise ValueError(f"invalid average accuracy in {path}: {line!r}") from error
            rows.append({"level": "overall", "term": "overall", "accuracy": accuracy})
            continue
        if ":" not in line:
            raise ValueError(f"invalid accuracy row in {path}: {line!r}")
        term, value = line.rsplit(":", 1)
        rows.append(
            {
                "level": section,
                "term": term.strip(),
                "accuracy": float(value.strip()),
            }
        )
    if not rows or sum(row["level"] == "overall" for row in rows) != 1:
        raise ValueError(f"report must contain exactly one average: {path}")
    return rows


def model_dirs(roots: list[Path], model_name: str) -> list[Path]:
    candidates = []
    for root in roots:
        for candidate in (root / model_name, root / "results" / model_name):
            if candidate.is_dir():
                resolved = candidate.resolve()
                if resolved not in candidates:
                    candidates.append(resolved)
    return candidates


def find_model_dir(roots: list[Path], model_name: str, required: bool) -> Path | None:
    candidates = model_dirs(roots, model_name)
    if len(candidates) > 1:
        raise RuntimeError(f"multiple result trees for {model_name}: {candidates}")
    if not candidates:
        if required:
            raise FileNotFoundError(f"no result tree found for {model_name}")
        return None
    return candidates[0]


def task_artifacts(model_dir: Path, task: str) -> tuple[Path, Path]:
    leaf = TASK_LEAVES[task]
    root = model_dir / "main" / "zero_shot" / "causal" / task / leaf
    return root / "best_temperature_report.txt", root / "predictions.json"


def add_task_rows(
    output: list[dict],
    *,
    model_dir: Path,
    run_name: str,
    checkpoint_label: str,
    iter_num: int,
    words_seen: int,
    mask_mode: str,
    task: str,
) -> None:
    report, predictions = task_artifacts(model_dir, task)
    if not report.is_file() or not predictions.is_file():
        raise FileNotFoundError(
            f"{task} requires report and predictions under {model_dir}"
        )
    parsed = parse_accuracy_report(report)
    if task == "blimp":
        term_count = sum(row["level"] == "linguistics_term" for row in parsed)
        if term_count != EXPECTED_BLIMP_TERMS:
            raise RuntimeError(
                f"expected {EXPECTED_BLIMP_TERMS} BLiMP terms, got {term_count}: {report}"
            )
    architecture, seed = run_metadata(run_name)
    for row in parsed:
        output.append(
            {
                "run_name": run_name,
                "architecture": architecture,
                "seed": seed,
                "checkpoint_label": checkpoint_label,
                "iter_num": iter_num,
                "words_seen": words_seen,
                "mask_mode": mask_mode,
                "task": task,
                **row,
                "report_path": str(report),
            }
        )


def read_position_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty position CSV: {path}")
    expected = list(range(len(rows)))
    observed = [int(row["loss_index"]) for row in rows]
    contexts = [int(row["context_length"]) for row in rows]
    if observed != expected or contexts != [index + 1 for index in expected]:
        raise ValueError(f"invalid loss-index/context-length sequence: {path}")
    return rows


def write_csv_exclusive(path: Path, rows: list[dict], fieldnames: tuple[str, ...]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-root", type=Path, action="append", required=True)
    parser.add_argument("--eval-results-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mask-mode",
        action="append",
        choices=("old", "embed", "random_count_matched"),
        dest="mask_modes",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="collate available artifacts without requiring all dev/mask outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    series_roots = [path.resolve() for path in args.series_root]
    eval_roots = [path.resolve() for path in args.eval_results_root]
    output_dir = args.output_dir.resolve()
    mask_modes = tuple(args.mask_modes or DEFAULT_MASK_MODES)
    plan_paths = sorted({path for root in series_roots for path in root.rglob("plan.json")})
    if not plan_paths:
        raise FileNotFoundError("no plan.json found under series roots")

    inventory_rows = []
    behavior_rows = []
    dev_rows = []
    position_rows = []
    plans: dict[str, list[dict]] = {}
    for plan_path in plan_paths:
        plan = load_json(plan_path).get("checkpoints")
        if not isinstance(plan, list) or len(plan) != 6:
            raise ValueError(f"expected six checkpoints in {plan_path}")
        run_name = plan[0]["run_name"]
        if run_name in plans:
            raise ValueError(f"duplicate series plan for {run_name}")
        plans[run_name] = plan
        architecture, seed = run_metadata(run_name)
        for item in plan:
            inventory_rows.append(
                {
                    **{key: item[key] for key in (
                        "run_name", "checkpoint_label", "checkpoint_path",
                        "checkpoint_filename", "checkpoint_role", "iter_num",
                        "tokens_seen", "words_seen", "sha256", "blimp_model_name",
                    )},
                    "architecture": architecture,
                    "seed": seed,
                }
            )
            checkpoint_dir = plan_path.parent / item["checkpoint_label"]
            dev_path = checkpoint_dir / "dev_loss.json"
            position_path = checkpoint_dir / "position_nll.csv"
            if dev_path.is_file() and position_path.is_file():
                dev = load_json(dev_path)
                if int(dev["iter_num"]) != item["iter_num"]:
                    raise RuntimeError(f"dev iter mismatch: {dev_path}")
                dev_rows.append(
                    {
                        "run_name": run_name,
                        "architecture": architecture,
                        "seed": seed,
                        "checkpoint_label": item["checkpoint_label"],
                        "checkpoint_role": item["checkpoint_role"],
                        "iter_num": item["iter_num"],
                        "tokens_seen": item["tokens_seen"],
                        "words_seen": item["words_seen"],
                        "mean_nll": dev["mean_nll"],
                        "scored_tokens": dev["scored_tokens"],
                        "val_bin_sha256": dev["val_bin_sha256_actual"],
                        "checkpoint_sha256": dev["checkpoint_sha256"],
                        "source_json": str(dev_path),
                    }
                )
                for row in read_position_rows(position_path):
                    position_rows.append(
                        {
                            "run_name": run_name,
                            "architecture": architecture,
                            "seed": seed,
                            "checkpoint_label": item["checkpoint_label"],
                            "iter_num": item["iter_num"],
                            "words_seen": item["words_seen"],
                            **{key: row[key] for key in (
                                "loss_index", "context_length", "nll_sum",
                                "token_count", "mean_nll",
                            )},
                        }
                    )
            elif not args.allow_incomplete:
                raise FileNotFoundError(f"missing dev pair under {checkpoint_dir}")

            model_dir = find_model_dir(
                eval_roots, item["blimp_model_name"], required=not args.allow_incomplete
            )
            if model_dir is not None:
                add_task_rows(
                    behavior_rows,
                    model_dir=model_dir,
                    run_name=run_name,
                    checkpoint_label=item["checkpoint_label"],
                    iter_num=item["iter_num"],
                    words_seen=item["words_seen"],
                    mask_mode="none",
                    task="blimp",
                )
                if item["checkpoint_label"] == "final":
                    for task in ("comps", "entity_tracking"):
                        add_task_rows(
                            behavior_rows,
                            model_dir=model_dir,
                            run_name=run_name,
                            checkpoint_label="final",
                            iter_num=item["iter_num"],
                            words_seen=item["words_seen"],
                            mask_mode="none",
                            task=task,
                        )

    for run_name, plan in plans.items():
        if "attnres" not in run_name:
            continue
        final = next(item for item in plan if item["checkpoint_label"] == "final")
        for mode in mask_modes:
            model_name = f"{run_name}--mask{mode}"
            model_dir = find_model_dir(
                eval_roots, model_name, required=not args.allow_incomplete
            )
            if model_dir is None:
                continue
            for task in TASK_LEAVES:
                add_task_rows(
                    behavior_rows,
                    model_dir=model_dir,
                    run_name=run_name,
                    checkpoint_label="final",
                    iter_num=final["iter_num"],
                    words_seen=final["words_seen"],
                    mask_mode=mode,
                    task=task,
                )

    inventory_rows.sort(key=lambda row: (row["seed"], row["architecture"], row["iter_num"]))
    dev_rows.sort(key=lambda row: (row["seed"], row["architecture"], row["iter_num"]))
    position_rows.sort(key=lambda row: (
        row["seed"], row["architecture"], row["iter_num"], int(row["loss_index"])
    ))
    behavior_rows.sort(key=lambda row: (
        row["seed"], row["architecture"], row["iter_num"], row["mask_mode"],
        row["task"], row["level"], row["term"],
    ))

    write_csv_exclusive(output_dir / "diag_checkpoint_inventory.csv", inventory_rows, (
        "run_name", "architecture", "seed", "checkpoint_label", "checkpoint_path",
        "checkpoint_filename", "checkpoint_role", "iter_num", "tokens_seen", "words_seen",
        "sha256", "blimp_model_name",
    ))
    write_csv_exclusive(output_dir / "diag_behavior_long.csv", behavior_rows, (
        "run_name", "architecture", "seed", "checkpoint_label", "iter_num", "words_seen",
        "mask_mode", "task", "level", "term", "accuracy", "report_path",
    ))
    write_csv_exclusive(output_dir / "diag_dev_loss.csv", dev_rows, (
        "run_name", "architecture", "seed", "checkpoint_label", "checkpoint_role",
        "iter_num", "tokens_seen", "words_seen", "mean_nll", "scored_tokens",
        "val_bin_sha256", "checkpoint_sha256", "source_json",
    ))
    write_csv_exclusive(output_dir / "diag_position_nll.csv", position_rows, (
        "run_name", "architecture", "seed", "checkpoint_label", "iter_num", "words_seen",
        "loss_index", "context_length", "nll_sum", "token_count", "mean_nll",
    ))
    print(
        f"wrote diagnosis CSVs: inventory={len(inventory_rows)} behavior={len(behavior_rows)} "
        f"dev={len(dev_rows)} position={len(position_rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
