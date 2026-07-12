#!/usr/bin/env python3
"""Build training_metadata.csv for scored runs from experiments.jsonl and W&B.

The score CSVs intentionally contain evaluation outputs only. This table is the
single source of truth for pretraining microbatch and validation-sampling
metadata, keyed by model/run name; build_all_runs.py joins it into all_runs.csv.
"""
import argparse
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "eval", "results")
FIELDS = [
    "model", "pretrain_batch_size", "pretrain_grad_accum", "tokens_per_update",
    "eval_batch_size", "eval_iters", "val_tokens_per_eval", "seed", "sampler_seed",
    "wandb_id", "metadata_source",
]


def scored_names():
    names = set()
    for filename in ("zero_shot.csv", "scale_up.csv", "glue.csv"):
        with open(os.path.join(RESULTS, filename), newline="") as f:
            names.update(row["model"] for row in csv.DictReader(f))
    return names


def load_experiments(paths):
    records = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = record.get("run_name")
                if name:
                    records[name] = record
    return records


def wandb_configs(project, names, experiments):
    if not project:
        return {}
    import wandb

    candidates = {}
    for run in wandb.Api().runs(project):
        if run.name in names:
            candidates.setdefault(run.name, []).append(run)
    selected = {}
    for name, runs in candidates.items():
        wanted_id = experiments.get(name, {}).get("wandb_id")
        match = next((run for run in runs if run.id == wanted_id), None)
        if match is None:
            finished = [run for run in runs if run.state == "finished"]
            match = (finished or runs)[0]
        selected[name] = (match.id, dict(match.config))
    return selected


def first(*values):
    return next((value for value in values if value is not None and value != ""), "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments", action="append",
        default=[os.path.join(ROOT, "results", "experiments.jsonl")],
        help="experiments.jsonl path; repeat for records from multiple boxes",
    )
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--output", default=os.path.join(RESULTS, "training_metadata.csv"))
    args = parser.parse_args()

    names = scored_names()
    experiments = load_experiments(args.experiments)
    wb = wandb_configs(args.wandb_project, names, experiments)
    rows = []
    for name in sorted(names):
        exp = experiments.get(name, {})
        wandb_id, config = wb.get(name, ("", {}))
        batch = first(config.get("batch_size"), exp.get("batch_size"))
        grad_accum = first(
            config.get("gradient_accumulation_steps"), exp.get("grad_accum")
        )
        block_size = first(config.get("block_size"), exp.get("block_size"))
        eval_batch = first(config.get("eval_batch_size"), batch)
        eval_iters = first(config.get("eval_iters"), 50 if batch != "" else "")
        tokens = first(exp.get("tokens_per_iter"))
        if tokens == "" and all(v != "" for v in (batch, grad_accum, block_size)):
            tokens = int(batch) * int(grad_accum) * int(block_size)
        val_tokens = ""
        if all(v != "" for v in (eval_batch, eval_iters, block_size)):
            val_tokens = int(eval_batch) * int(eval_iters) * int(block_size)
        source = "+".join(
            part for part, present in (
                ("experiments", bool(exp)), ("wandb", name in wb)
            ) if present
        )
        rows.append({
            "model": name,
            "pretrain_batch_size": batch,
            "pretrain_grad_accum": grad_accum,
            "tokens_per_update": tokens,
            "eval_batch_size": eval_batch,
            "eval_iters": eval_iters,
            "val_tokens_per_eval": val_tokens,
            "seed": first(config.get("seed"), exp.get("seed")),
            "sampler_seed": first(config.get("sampler_seed"), exp.get("sampler_seed")),
            "wandb_id": first(exp.get("wandb_id"), wandb_id),
            "metadata_source": source,
        })

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}: {len(rows)} scored runs")


if __name__ == "__main__":
    main()
