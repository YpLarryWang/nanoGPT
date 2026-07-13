#!/usr/bin/env python3
"""Validate BabyLM scoreboard formulas, schemas, and source/all_runs agreement."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
ERRORS: list[str] = []


def rows(path: Path, key_fields: tuple[str, ...] | None = None) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        result = list(csv.DictReader(handle))
    if key_fields:
        keys = [tuple(row.get(field, "") for field in key_fields) for row in result]
    else:
        keys = [row.get("model") or row.get("run_name") for row in result]
    duplicates = sorted({key for key in keys if key and keys.count(key) > 1})
    if duplicates:
        ERRORS.append(f"{path.name}: duplicate keys {duplicates}")
    return result


def numeric(row: dict[str, str], fields: tuple[str, ...]) -> list[float] | None:
    values = [row.get(field, "") for field in fields]
    if not any(values):
        return None
    if not all(values):
        return None
    try:
        return [float(value) for value in values]
    except ValueError:
        return None


def check_aggregate(
    path: Path, row: dict[str, str], inputs: tuple[str, ...], output: str,
    excluded: str | None = None, tolerance: float = 0.011,
) -> None:
    values = numeric(row, inputs)
    if values is None:
        return
    included = [value for field, value in zip(inputs, values) if field != excluded]
    expected = sum(included) / len(included)
    try:
        actual = float(row[output])
    except (KeyError, ValueError):
        ERRORS.append(f"{path.name}:{row.get('model') or row.get('run_name')}: missing {output}")
        return
    if abs(expected - actual) > tolerance:
        ERRORS.append(
            f"{path.name}:{row.get('model') or row.get('run_name')}: "
            f"{output}={actual:.2f}, recomputed={expected:.2f}"
        )


def main() -> None:
    glue_path = RESULTS / "glue.csv"
    glue_rows = rows(glue_path)
    glue_inputs = ("boolq_acc", "multirc_acc", "rte_acc", "wsc_acc", "mrpc_f1", "qqp_f1", "mnli_acc")
    for row in glue_rows:
        check_aggregate(glue_path, row, glue_inputs, "macro7")
        check_aggregate(glue_path, row, glue_inputs, "macro6", excluded="wsc_acc")

    zero_inputs = ("blimp", "supplement", "ewok", "entity_tracking", "comps")
    for name in ("zero_shot.csv", "scale_up.csv"):
        path = RESULTS / name
        for row in rows(path):
            check_aggregate(path, row, zero_inputs, "avg5")

    all_path = RESULTS / "all_runs.csv"
    all_rows = rows(all_path)
    for row in all_rows:
        check_aggregate(all_path, row, ("blimp", "supplement", "ewok", "comps"), "reliable4")

    # Every scored source field must be reproduced exactly in all_runs.csv.
    all_by_name = {row["run_name"]: row for row in all_rows}
    mappings = {
        "glue.csv": {
            "boolq_acc": "boolq", "multirc_acc": "multirc", "rte_acc": "rte",
            "wsc_acc": "wsc", "mrpc_f1": "mrpc", "qqp_f1": "qqp", "mnli_acc": "mnli",
            "macro7": "macro7", "macro6": "macro6",
        },
        "zero_shot.csv": {field: field for field in
                          ("blimp", "supplement", "ewok", "entity_tracking", "comps",
                           "global_piqa_parallel", "global_piqa_nonparallel", "avg5",
                           "reading_eye", "reading_selfpaced")},
        "scale_up.csv": {field: field for field in
                         ("blimp", "supplement", "ewok", "entity_tracking", "comps",
                          "global_piqa_parallel", "global_piqa_nonparallel", "avg5",
                          "reading_eye", "reading_selfpaced")},
    }
    for filename, mapping in mappings.items():
        for source in rows(RESULTS / filename):
            target = all_by_name.get(source["model"])
            if target is None:
                ERRORS.append(f"all_runs.csv: missing source model {source['model']} from {filename}")
                continue
            for source_field, target_field in mapping.items():
                value = source.get(source_field, "")
                if value and target.get(target_field, "") != value:
                    ERRORS.append(
                        f"all_runs.csv:{source['model']}:{target_field}={target.get(target_field)!r}, "
                        f"expected {value!r} from {filename}:{source_field}"
                    )

    fast_path = RESULTS / "fast_zero_shot.csv"
    if fast_path.exists():
        seen: set[tuple[str, str, str]] = set()
        required = ("blimp", "supplement", "ewok", "entity_tracking",
                    "global_piqa_parallel", "global_piqa_nonparallel",
                    "reading_eye", "reading_selfpaced")
        for row in rows(fast_path, key_fields=("model", "revision", "backend")):
            key = (row["model"], row["revision"], row["backend"])
            if key in seen:
                ERRORS.append(f"fast_zero_shot.csv: duplicate key {key}")
            seen.add(key)
            for field in required:
                try:
                    float(row[field])
                except (KeyError, ValueError):
                    ERRORS.append(f"fast_zero_shot.csv:{key}: missing/non-numeric {field}")

    if ERRORS:
        print("result validation failed:", file=sys.stderr)
        for error in ERRORS:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"result validation passed: {len(all_rows)} unified runs")


if __name__ == "__main__":
    main()
