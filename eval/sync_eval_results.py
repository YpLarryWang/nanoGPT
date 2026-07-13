#!/usr/bin/env python3
"""Import BabyLM eval outputs directly into the project scoreboards.

This is the only supported bridge from ``babylm-eval/strict/results`` to the
CSV files in ``eval/results``.  It deliberately selects metrics by task name
and metric key; no result is selected by position and no aggregate is entered
by hand.

Examples (run from nanoGPT):

  python eval/sync_eval_results.py MODEL --glue
  python eval/sync_eval_results.py MODEL --full --backend causal
  python eval/sync_eval_results.py MODEL --fast --revision chck_10M
  python eval/sync_eval_results.py MODEL --fast --all-revisions

GLUE is written to glue.csv. Full zero-shot updates the existing row in
zero_shot.csv and/or scale_up.csv. Fast checkpoint results are written to the
checkpoint-granularity fast_zero_shot.csv. Any GLUE/full change automatically
rebuilds all_runs.csv, and every write is followed by validate_results.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
DEFAULT_EVAL_REPO = Path("/media/volume/yupei-data/repo/babylm-eval/strict")

GLUE_TASK_METRIC = {
    "boolq": "accuracy",
    "multirc": "accuracy",
    "rte": "accuracy",
    "wsc": "accuracy",
    "mrpc": "f1",
    "qqp": "f1",
    "mnli": "accuracy",
}
GLUE_COLUMNS = {
    "boolq": "boolq_acc",
    "multirc": "multirc_acc",
    "rte": "rte_acc",
    "wsc": "wsc_acc",
    "mrpc": "mrpc_f1",
    "qqp": "qqp_f1",
    "mnli": "mnli_acc",
}
GLUE_SCORE_COLUMNS = list(GLUE_COLUMNS.values())

ZERO_REPORTS_FULL = {
    "blimp": ("blimp", "blimp_filtered"),
    "supplement": ("blimp", "supplement_filtered"),
    "ewok": ("ewok", "ewok_filtered"),
    "entity_tracking": ("entity_tracking", "entity_tracking"),
    "comps": ("comps", "comps"),
    "global_piqa_parallel": ("global_piqa_parallel", "global_piqa_parallel"),
    "global_piqa_nonparallel": ("global_piqa_nonparallel", "global_piqa_nonparallel"),
}
ZERO_REPORTS_FAST = {
    "blimp": ("blimp", "blimp_fast"),
    "supplement": ("blimp", "supplement_fast"),
    "ewok": ("ewok", "ewok_fast"),
    "entity_tracking": ("entity_tracking", "entity_tracking_fast"),
    "global_piqa_parallel": ("global_piqa_parallel", "global_piqa_parallel"),
    "global_piqa_nonparallel": ("global_piqa_nonparallel", "global_piqa_nonparallel"),
}
FULL_SCORE_COLUMNS = [
    "blimp", "supplement", "ewok", "entity_tracking", "comps",
    "global_piqa_parallel", "global_piqa_nonparallel",
    "avg5", "reading_eye", "reading_selfpaced",
]
FAST_FIELDS = [
    "model", "revision", "backend", "track", "blimp", "supplement", "ewok",
    "entity_tracking", "global_piqa_parallel", "global_piqa_nonparallel",
    "reading_eye", "reading_selfpaced", "source",
]


def score(value: float) -> str:
    return f"{value:.2f}"


def parse_key_value_metrics(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"missing result file: {path}")
    metrics: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        try:
            value = float(raw.strip())
        except ValueError:
            continue
        metrics[key] = value
    return metrics


def parse_average_accuracy(path: Path) -> float:
    """Read only the explicitly labelled aggregate from a zero-shot report."""
    if not path.is_file():
        raise FileNotFoundError(f"missing zero-shot report: {path}")
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "### AVERAGE ACCURACY":
            continue
        for candidate in lines[index + 1 :]:
            candidate = candidate.strip()
            if not candidate:
                continue
            try:
                return float(candidate)
            except ValueError as exc:
                raise ValueError(f"invalid average accuracy in {path}: {candidate!r}") from exc
    raise ValueError(f"no '### AVERAGE ACCURACY' section in {path}")


def parse_reading_report(path: Path) -> tuple[float, float]:
    if not path.is_file():
        raise FileNotFoundError(f"missing reading report: {path}")
    values: dict[str, float] = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"(EYE TRACKING|SELF-PACED READING) SCORE:\s*([-+0-9.eE]+)", line.strip())
        if match:
            values[match.group(1)] = float(match.group(2))
    required = {"EYE TRACKING", "SELF-PACED READING"}
    if values.keys() != required:
        raise ValueError(f"reading report has keys {sorted(values)}, expected {sorted(required)}: {path}")
    return values["EYE TRACKING"], values["SELF-PACED READING"]


def collect_glue(eval_repo: Path, model: str, revision: str) -> dict[str, str]:
    base = eval_repo / "results" / model / revision / "finetune"
    raw_scores: list[float] = []
    updates: dict[str, str] = {}
    for task, metric_key in GLUE_TASK_METRIC.items():
        path = base / task / "results.txt"
        metrics = parse_key_value_metrics(path)
        if metric_key not in metrics:
            raise ValueError(f"{path} has {sorted(metrics)}, missing required {metric_key!r}")
        value = metrics[metric_key]
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"expected fractional {task}:{metric_key} in [0,1], got {value}")
        percentage = value * 100.0
        raw_scores.append(percentage)
        updates[GLUE_COLUMNS[task]] = score(percentage)
    updates["macro7"] = score(sum(raw_scores) / len(raw_scores))
    wsc = float(updates["wsc_acc"])
    # Use the raw task values except WSC; formatting happens only after aggregation.
    raw_wsc = parse_key_value_metrics(base / "wsc" / "results.txt")["accuracy"] * 100.0
    updates["macro6"] = score((sum(raw_scores) - raw_wsc) / 6)
    assert abs(wsc - raw_wsc) < 0.01
    return updates


def zero_report_path(base: Path, parts: tuple[str, str]) -> Path:
    return base / parts[0] / parts[1] / "best_temperature_report.txt"


def collect_zero_shot(
    eval_repo: Path, model: str, backend: str, revision: str, fast: bool
) -> dict[str, str]:
    base = eval_repo / "results" / model / revision / "zero_shot" / backend
    reports = ZERO_REPORTS_FAST if fast else ZERO_REPORTS_FULL
    updates = {column: score(parse_average_accuracy(zero_report_path(base, parts)))
               for column, parts in reports.items()}
    eye, selfpaced = parse_reading_report(base / "reading" / "report.txt")
    updates["reading_eye"] = score(eye)
    updates["reading_selfpaced"] = score(selfpaced)
    if not fast:
        core = [float(updates[column]) for column in
                ("blimp", "supplement", "ewok", "entity_tracking", "comps")]
        updates["avg5"] = score(sum(core) / 5)
    return updates


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def insert_after(fields: list[str], anchor: str, additions: list[str]) -> list[str]:
    result = [field for field in fields if field not in additions]
    index = result.index(anchor) + 1
    result[index:index] = additions
    return result


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]], dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        tmp = Path(handle.name)
    tmp.replace(path)


def find_row(rows: list[dict[str, str]], model: str) -> dict[str, str] | None:
    matches = [row for row in rows if row.get("model") == model]
    if len(matches) > 1:
        raise ValueError(f"duplicate model rows for {model}")
    return matches[0] if matches else None


def update_full_tables(
    model: str, updates: dict[str, str], metadata_from: str | None,
    source_note: str | None, dry_run: bool,
) -> list[Path]:
    changed: list[Path] = []
    tables = [RESULTS / "zero_shot.csv", RESULTS / "scale_up.csv"]
    loaded = [(path, *read_csv(path)) for path in tables]
    targets = [(path, fields, rows) for path, fields, rows in loaded if find_row(rows, model)]

    if not targets and metadata_from:
        templates = [(path, fields, rows, find_row(rows, metadata_from))
                     for path, fields, rows in loaded]
        templates = [item for item in templates if item[3] is not None]
        if len(templates) > 1:
            # Some original ablations intentionally appear in both tables; the
            # scale-up schema is the canonical home for new formal variants.
            templates = [item for item in templates if item[0].name == "scale_up.csv"]
        if len(templates) != 1:
            raise ValueError(
                f"--metadata-from {metadata_from!r} must match exactly one source table; "
                f"matched {len(templates)}"
            )
        path, fields, rows, template = templates[0]
        new_row = dict(template or {})
        new_row["model"] = model
        for column in FULL_SCORE_COLUMNS:
            new_row[column] = ""
        rows.append(new_row)
        targets = [(path, fields, rows)]

    if not targets and not metadata_from:
        path = RESULTS / "scale_up.csv"
        fields, rows = read_csv(path)
        rows.append(training_source_row(model, fields))
        targets = [(path, fields, rows)]

    if not targets:
        raise ValueError(
            f"{model!r} is absent from zero_shot.csv and scale_up.csv; add --metadata-from BASE_MODEL "
            "to clone the non-score metadata in the same command"
        )

    for path, fields, rows in targets:
        fields = insert_after(fields, "comps", ["global_piqa_parallel", "global_piqa_nonparallel"])
        row = find_row(rows, model)
        assert row is not None
        row.update(updates)
        if source_note:
            row["source"] = source_note
        write_csv(path, fields, rows, dry_run)
        changed.append(path)
    return changed


def infer_train_words(model: str) -> str:
    if "bl100m-" in model:
        return "100M"
    if "bl10m-" in model:
        return "10M"
    raise ValueError(f"cannot infer train_words from model name: {model}")


def infer_arch(model: str) -> str:
    if re.search(r"(?:^|-)d\d+L\d+", model):
        return "RMS+SwiGLU(8/3)+RoPE" + ("+gate" if "-gate" in model else "")
    raise ValueError(f"cannot infer architecture from model name: {model}")


def latest_experiment(model: str) -> dict[str, object]:
    path = ROOT / "results" / "experiments.jsonl"
    matches: list[dict[str, object]] = []
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("run_name") == model:
                matches.append(record)
    return matches[-1] if matches else {}


def checkpoint_config(model: str) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    path = ROOT / "out-babylm" / model / "ckpt.pt"
    if not path.is_file():
        return {}, {}, {}
    try:
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.6
        import torch
        checkpoint = torch.load(path, map_location="cpu")
    return (
        dict(checkpoint.get("model_args", {})),
        dict(checkpoint.get("config", {})),
        checkpoint,
    )


def training_source_row(model: str, fields: list[str]) -> dict[str, str]:
    """Build a scale_up.csv metadata row without inventing score values."""
    experiment = latest_experiment(model)
    model_args, config, checkpoint = checkpoint_config(model)
    if not experiment and not model_args:
        raise ValueError(
            f"{model!r} has no source-table row and no local training metadata/checkpoint; "
            "use --metadata-from BASE_MODEL"
        )
    row = {field: "" for field in fields}
    row.update({
        "model": model,
        "n_embd": str(experiment.get("n_embd") or model_args.get("n_embd") or ""),
        "n_layer": str(experiment.get("n_layer") or model_args.get("n_layer") or ""),
        "params_M": score(float(experiment["params_M"])) if experiment.get("params_M") else "",
        "dropout": str(config.get("dropout", model_args.get("dropout", ""))),
        "sampler": str(config.get("sampler", "")),
        "train_words": infer_train_words(model),
        "best_val_loss": str(
            experiment.get("best_val_loss") or checkpoint.get("best_val_loss") or ""
        ),
        "source": f"ours(measured;structured import;{date.today().isoformat()})",
    })
    missing = [field for field in ("n_embd", "n_layer", "params_M", "dropout", "sampler")
               if not row[field]]
    if missing:
        raise ValueError(f"cannot create scale_up.csv row for {model}; missing metadata {missing}")
    return row


def update_glue_table(
    model: str, updates: dict[str, str], metadata_from: str | None,
    source_note: str | None, dry_run: bool,
) -> Path:
    path = RESULTS / "glue.csv"
    fields, rows = read_csv(path)
    row = find_row(rows, model)
    if row is None and metadata_from:
        template = find_row(rows, metadata_from)
        if template is None:
            raise ValueError(f"GLUE metadata template not found: {metadata_from}")
        row = dict(template)
        row["model"] = model
        for column in GLUE_SCORE_COLUMNS + ["macro7", "macro6"]:
            row[column] = ""
        rows.append(row)
    if row is None:
        # Prefer metadata already recorded for this run in the unified table.
        _, all_rows = read_csv(RESULTS / "all_runs.csv")
        unified = next((item for item in all_rows if item.get("run_name") == model), None)
        if unified is None:
            arch = infer_arch(model)
            train_words = infer_train_words(model)
        else:
            arch = unified.get("arch", "")
            train_words = unified.get("train_words") or infer_train_words(model)
        row = {field: "" for field in fields}
        row.update({
            "model": model,
            "arch": arch,
            "train_words": train_words,
        })
        rows.append(row)
    row.update(updates)
    if source_note:
        row["source"] = source_note
    elif not row.get("source"):
        row["source"] = f"ours(measured;structured import;{date.today().isoformat()})"
    write_csv(path, fields, rows, dry_run)
    return path


def revision_sort_key(revision: str) -> tuple[int, int | str]:
    if revision == "main":
        return (0, 0)
    match = re.fullmatch(r"chck_(\d+)M", revision)
    return (1, int(match.group(1))) if match else (2, revision)


def discover_revisions(eval_repo: Path, model: str, backend: str) -> list[str]:
    model_dir = eval_repo / "results" / model
    if not model_dir.is_dir():
        raise FileNotFoundError(f"missing result directory: {model_dir}")
    # A main full-eval directory is not a fast revision. Require the labelled
    # BLiMP-fast report instead of merely checking for a zero_shot directory.
    revisions = [
        path.name for path in model_dir.iterdir()
        if (path / "zero_shot" / backend / "blimp" / "blimp_fast" /
            "best_temperature_report.txt").is_file()
    ]
    return sorted(revisions, key=revision_sort_key)


def update_fast_table(
    model: str, backend: str, revision_updates: list[tuple[str, dict[str, str]]],
    source_note: str | None, dry_run: bool,
) -> Path:
    path = RESULTS / "fast_zero_shot.csv"
    fields, rows = read_csv(path)
    if not fields:
        fields = list(FAST_FIELDS)
    else:
        fields = insert_after(fields, "entity_tracking", ["global_piqa_parallel", "global_piqa_nonparallel"])
    for revision, updates in revision_updates:
        matches = [row for row in rows if (row.get("model"), row.get("revision"), row.get("backend"))
                   == (model, revision, backend)]
        if len(matches) > 1:
            raise ValueError(f"duplicate fast row: {(model, revision, backend)}")
        if matches:
            row = matches[0]
        else:
            row = {field: "" for field in fields}
            row.update({
                "model": model,
                "revision": revision,
                "backend": backend,
                "track": "strict-small" if "bl10m-" in model else "strict",
            })
            rows.append(row)
        row.update(updates)
        row["source"] = source_note or row.get("source") or \
            f"ours(measured;structured import;{date.today().isoformat()})"
    rows.sort(key=lambda row: (row.get("model", ""), revision_sort_key(row.get("revision", "")),
                               row.get("backend", "")))
    write_csv(path, fields, rows, dry_run)
    return path


def print_updates(label: str, updates: dict[str, str]) -> None:
    rendered = ", ".join(f"{key}={value}" for key, value in updates.items())
    print(f"{label}: {rendered}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--csv-model",
                        help="scoreboard model name when it differs from the eval result directory")
    parser.add_argument("--eval-repo", type=Path, default=DEFAULT_EVAL_REPO)
    parser.add_argument("--backend", default="causal")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--glue", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--all", action="store_true", help="import GLUE, full, and fast outputs")
    parser.add_argument("--all-revisions", action="store_true",
                        help="with --fast, import every revision found under results/MODEL")
    parser.add_argument("--metadata-from",
                        help="clone non-score CSV metadata when MODEL has no source row yet")
    parser.add_argument("--source-note", help="replace the source/provenance cell")
    parser.add_argument("--dry-run", action="store_true", help="parse and print without writing")
    args = parser.parse_args()

    if args.all:
        args.glue = args.full = args.fast = True
    if not (args.glue or args.full or args.fast):
        parser.error("select at least one of --glue, --full, --fast, or --all")
    if args.all_revisions and not args.fast:
        parser.error("--all-revisions requires --fast")

    csv_model = args.csv_model or args.model
    changed: list[Path] = []
    if args.glue:
        updates = collect_glue(args.eval_repo, args.model, args.revision)
        print_updates("GLUE", updates)
        changed.append(update_glue_table(
            csv_model, updates, args.metadata_from, args.source_note, args.dry_run
        ))

    if args.full:
        updates = collect_zero_shot(
            args.eval_repo, args.model, args.backend, args.revision, fast=False
        )
        print_updates("full zero-shot", updates)
        changed.extend(update_full_tables(
            csv_model, updates, args.metadata_from, args.source_note, args.dry_run
        ))

    if args.fast:
        revisions = discover_revisions(args.eval_repo, args.model, args.backend) \
            if args.all_revisions else [args.revision]
        revision_updates = []
        for revision in revisions:
            updates = collect_zero_shot(
                args.eval_repo, args.model, args.backend, revision, fast=True
            )
            print_updates(f"fast zero-shot {revision}", updates)
            revision_updates.append((revision, updates))
        changed.append(update_fast_table(
            csv_model, args.backend, revision_updates, args.source_note, args.dry_run
        ))

    if args.dry_run:
        print("dry-run: no files written")
        return

    if args.glue or args.full:
        subprocess.run([sys.executable, str(ROOT / "eval" / "build_all_runs.py")], check=True)
        changed.append(RESULTS / "all_runs.csv")
    subprocess.run([sys.executable, str(ROOT / "eval" / "validate_results.py")], check=True)
    print("updated: " + ", ".join(str(path.relative_to(ROOT)) for path in dict.fromkeys(changed)))


if __name__ == "__main__":
    main()
