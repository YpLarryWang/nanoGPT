#!/usr/bin/env python3
"""Consolidate the per-experiment result CSVs into ONE master table: all_runs.csv.

The three source CSVs stay the source of truth (one Google-Sheets tab each):
  zero_shot.csv  - 33M architecture-ablation zero-shot + baselines
  scale_up.csv   - scale-up zero-shot (depth/width/dropout/gate/optimizer)
  glue.csv       - GLUE fine-tuning

This emits one row per run with a unified, explicit schema: metadata
(optimizer, gate, arch, size, dropout) + zero-shot metrics + best_val_loss +
GLUE metrics. A blank cell means that run was not measured on that suite. Runs
that appear in several source CSVs are merged into a single row.

Re-run whenever a source CSV changes:  python eval/build_all_runs.py
"""
import csv
import os

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
COLS = [
    "run_name", "optimizer", "gate", "arch", "train_words", "n_embd", "n_layer",
    "params_M", "dropout", "sampler",
    "pretrain_batch_size", "pretrain_grad_accum", "tokens_per_update",
    "eval_batch_size", "eval_iters", "val_tokens_per_eval", "seed", "sampler_seed",
    "wandb_id", "metadata_source",
    "blimp", "supplement", "ewok", "entity_tracking", "comps", "avg5", "reliable4",
    "reading_eye", "reading_selfpaced", "best_val_loss",
    "boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli", "macro7", "macro6",
    "eval_tok", "source",
]

runs = {}
order = []

def row_for(name):
    if name not in runs:
        r = {c: "" for c in COLS}
        r["run_name"] = name
        r["optimizer"] = "muon" if name.startswith("muon-") else "adamw"
        r["gate"] = "yes" if "-gate" in name else "no"
        runs[name] = r
        order.append(name)
    return runs[name]

def load(fname, mapping):
    path = os.path.join(HERE, fname)
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        for src in csv.DictReader(f):
            r = row_for(src["model"])
            for dst, col in mapping.items():
                v = src.get(col, "")
                if v != "":
                    r[dst] = v

load("zero_shot.csv", {
    "arch": "arch", "train_words": "train_words", "params_M": "params_M",
    "blimp": "blimp", "supplement": "supplement", "ewok": "ewok",
    "entity_tracking": "entity_tracking", "comps": "comps", "avg5": "avg5",
    "reading_eye": "reading_eye", "reading_selfpaced": "reading_selfpaced",
    "eval_tok": "eval_tok", "source": "source",
})
load("scale_up.csv", {
    "train_words": "train_words", "n_embd": "n_embd", "n_layer": "n_layer",
    "params_M": "params_M", "dropout": "dropout", "sampler": "sampler",
    "blimp": "blimp", "supplement": "supplement", "ewok": "ewok",
    "entity_tracking": "entity_tracking", "comps": "comps", "avg5": "avg5",
    "reading_eye": "reading_eye", "reading_selfpaced": "reading_selfpaced",
    "best_val_loss": "best_val_loss", "source": "source",
})
load("glue.csv", {
    "arch": "arch", "train_words": "train_words",
    "boolq": "boolq_acc", "multirc": "multirc_acc", "rte": "rte_acc",
    "wsc": "wsc_acc", "mrpc": "mrpc_f1", "qqp": "qqp_f1", "mnli": "mnli_acc",
    "macro7": "macro7", "macro6": "macro6",
})
load("training_metadata.csv", {
    "pretrain_batch_size": "pretrain_batch_size",
    "pretrain_grad_accum": "pretrain_grad_accum",
    "tokens_per_update": "tokens_per_update",
    "eval_batch_size": "eval_batch_size", "eval_iters": "eval_iters",
    "val_tokens_per_eval": "val_tokens_per_eval", "seed": "seed",
    "sampler_seed": "sampler_seed", "wandb_id": "wandb_id",
    "metadata_source": "metadata_source",
})

for r in runs.values():
    # reliable4 = mean(BLiMP, supplement, EWoK, COMPS) -- the fair metric (drops noisy entity)
    try:
        vs = [float(r[k]) for k in ("blimp", "supplement", "ewok", "comps")]
        r["reliable4"] = f"{sum(vs) / 4:.2f}"
    except ValueError:
        pass
    # scale-up / muon rows carry the winner arch; label it if zero_shot.csv didn't
    if not r["arch"] and r["n_layer"]:
        r["arch"] = "RMS+SwiGLU(8/3)+RoPE" + ("+gate" if r["gate"] == "yes" else "")

def sort_key(name):
    r = runs[name]
    is_base = 1 if "baseline" in name.lower() else 0
    track = 0 if r["train_words"] == "100M" else 1
    try:
        p = int(float(r["params_M"]))
    except ValueError:
        p = 0
    return (is_base, track, p, name)

with open(os.path.join(HERE, "all_runs.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n")
    w.writeheader()
    for name in sorted(order, key=sort_key):
        w.writerow(runs[name])

print(f"wrote all_runs.csv: {len(order)} runs")
