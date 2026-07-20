#!/usr/bin/env python3
"""Validate, summarize, and plot the frozen diagnosis supplement."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np

from diag_make_figures import interpolate_on_nonincreasing_x, normalized_log_auc, pava_nonincreasing


SEEDS = (1337, 1338, 1339)
MASK_SEEDS = (20260718, 20260719, 20260720, 20260721, 20260722)
METRICS = {
    "overall_blimp": ("blimp", "overall", "overall"),
    "filler_gap_dependency": ("blimp", "linguistics_term", "filler_gap_dependency"),
}
COLORS = {"baseline": "#6b7280", "attnres": "#087e8b"}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def unique_index(rows: list[dict], fields: tuple[str, ...], label: str) -> dict:
    output = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in fields)
        if key in output:
            raise ValueError(f"duplicate {label} key: {key}")
        output[key] = row
    return output


def corpus_for(row: dict) -> str:
    return row.get("corpus") or ("100m" if row["run_name"].startswith("bl100m-") else "10m")


def behavior_index(rows: list[dict]) -> dict:
    return unique_index(
        rows,
        (
            "corpus", "seed", "architecture", "checkpoint_label", "mask_mode",
            "mask_seed", "task", "level", "term",
        ),
        "behavior",
    )


def dev_index(rows: list[dict]) -> dict:
    normalized = [{**row, "corpus": corpus_for(row)} for row in rows]
    return unique_index(normalized, ("corpus", "seed", "architecture", "checkpoint_label"), "dev")


def metric_value(index: dict, corpus: str, seed: int, architecture: str, label: str, spec) -> float:
    task, level, term = spec
    key = (corpus, str(seed), architecture, label, "none", "", task, level, term)
    if key not in index:
        raise KeyError(f"missing behavior row: {key}")
    return float(index[key]["accuracy"])


def paired_series(behavior: list[dict], dev: list[dict], corpus: str, seed: int, spec) -> dict:
    bidx = behavior_index(behavior)
    didx = dev_index(dev)
    labels = sorted(
        {
            key[3]
            for key in didx
            if key[0] == corpus and key[1] == str(seed) and key[2] == "baseline"
            and (key[3] != "final" or corpus == "10m")
        },
        key=lambda label: (
            float(didx[(corpus, str(seed), "baseline", label)]["words_seen"]),
            label == "final",
            label,
        ),
    )
    expected = 19 if corpus == "10m" else 9
    if len(labels) != expected:
        raise RuntimeError(f"{corpus} seed {seed} has {len(labels)} points, expected {expected}")
    baseline_dev = [didx[(corpus, str(seed), "baseline", label)] for label in labels]
    attnres_dev = [didx[(corpus, str(seed), "attnres", label)] for label in labels]
    words = np.array([float(row["words_seen"]) for row in baseline_dev])
    attn_words = np.array([float(row["words_seen"]) for row in attnres_dev])
    if not np.array_equal(words, attn_words):
        raise RuntimeError(f"paired words differ: corpus={corpus} seed={seed}")
    return {
        "labels": labels,
        "words": words,
        "iters": np.array([int(row["iter_num"]) for row in baseline_dev]),
        "baseline_accuracy": np.array([
            metric_value(bidx, corpus, seed, "baseline", label, spec) for label in labels
        ]),
        "attnres_accuracy": np.array([
            metric_value(bidx, corpus, seed, "attnres", label, spec) for label in labels
        ]),
        "baseline_nll": np.array([float(row["mean_nll"]) for row in baseline_dev]),
        "attnres_nll": np.array([float(row["mean_nll"]) for row in attnres_dev]),
    }


def longest_positive_window(labels: list[str], deltas: np.ndarray) -> tuple[str, str, int]:
    best = ("", "", 0)
    start = None
    for index, value in enumerate(np.append(deltas, 0.0)):
        if value > 0 and start is None:
            start = index
        elif value <= 0 and start is not None:
            length = index - start
            if length > best[2]:
                best = (labels[start], labels[index - 1], length)
            start = None
    return best


def trajectory_summaries(behavior: list[dict], dev: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    plotting = {}
    for metric, spec in METRICS.items():
        plotting[metric] = {}
        for seed in SEEDS:
            series = paired_series(behavior, dev, "10m", seed, spec)
            plotting[metric][seed] = series
            for sensitivity, keep in (
                ("all_points", np.ones(len(series["words"]), dtype=bool)),
                ("exclude_1m", np.array([label != "1M" for label in series["labels"]])),
                ("post_warmup", series["iters"] >= 40),
            ):
                labels = [label for label, selected in zip(series["labels"], keep, strict=True) if selected]
                words = series["words"][keep]
                baseline_acc = series["baseline_accuracy"][keep]
                attnres_acc = series["attnres_accuracy"][keep]
                baseline_nll = series["baseline_nll"][keep]
                attnres_nll = series["attnres_nll"][keep]
                deltas = attnres_acc - baseline_acc
                iso = pava_nonincreasing(baseline_nll)
                matched = []
                matched_words = []
                for word, target, accuracy in zip(words, attnres_nll, attnres_acc, strict=True):
                    comparator = interpolate_on_nonincreasing_x(float(target), iso, baseline_acc)
                    if comparator is not None:
                        matched_words.append(word)
                        matched.append(float(accuracy - comparator))
                maximum = int(np.argmax(deltas))
                window = longest_positive_window(labels, deltas)
                final_outside = int(
                    "final" in labels
                    and not (float(np.min(iso)) <= float(attnres_nll[labels.index("final")]) <= float(np.max(iso)))
                )
                rows.append({
                    "metric": metric,
                    "seed": seed,
                    "sensitivity": sensitivity,
                    "n_points": len(words),
                    "log_exposure_auc_pp": normalized_log_auc(words, deltas),
                    "largest_observed_label": labels[maximum],
                    "largest_observed_words": int(words[maximum]),
                    "largest_observed_delta_pp": float(deltas[maximum]),
                    "positive_window_start": window[0],
                    "positive_window_end": window[1],
                    "positive_window_points": window[2],
                    "loss_matched_mean_delta_pp": float(np.mean(matched)) if matched else math.nan,
                    "loss_matched_auc_pp": (
                        normalized_log_auc(matched_words, matched) if len(matched) >= 2 else math.nan
                    ),
                    "n_loss_matched": len(matched),
                    "pava_applied": int(not np.array_equal(iso, baseline_nll)),
                    "final_outside_baseline_overlap": final_outside,
                })
    for seed in SEEDS:
        series = paired_series(behavior, dev, "10m", seed, METRICS["overall_blimp"])
        delta = series["baseline_nll"] - series["attnres_nll"]
        maximum = int(np.argmax(delta))
        rows.append({
            "metric": "dev_nll",
            "seed": seed,
            "sensitivity": "all_points",
            "n_points": len(delta),
            "log_exposure_auc_pp": normalized_log_auc(series["words"], delta),
            "largest_observed_label": series["labels"][maximum],
            "largest_observed_words": int(series["words"][maximum]),
            "largest_observed_delta_pp": float(delta[maximum]),
            "positive_window_start": longest_positive_window(series["labels"], delta)[0],
            "positive_window_end": longest_positive_window(series["labels"], delta)[1],
            "positive_window_points": longest_positive_window(series["labels"], delta)[2],
            "loss_matched_mean_delta_pp": "",
            "loss_matched_auc_pp": "",
            "n_loss_matched": "",
            "pava_applied": "",
            "final_outside_baseline_overlap": "",
        })
    return rows, plotting


def masking_tables(behavior: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    index = behavior_index(behavior)
    terms = sorted({
        key[-1] for key in index
        if key[:5] == ("10m", key[1], "attnres", "final", "none")
        and key[6:8] == ("blimp", "linguistics_term")
    })
    if len(terms) != 13:
        raise RuntimeError(f"expected 13 native BLiMP terms, got {len(terms)}")
    draws = []
    for seed in SEEDS:
        for term in terms:
            common = ("10m", str(seed))
            u = float(index[common + ("attnres", "final", "none", "", "blimp", "linguistics_term", term)]["accuracy"])
            b = float(index[common + ("baseline", "final", "none", "", "blimp", "linguistics_term", term)]["accuracy"])
            o = float(index[common + ("attnres", "final", "old", "", "blimp", "linguistics_term", term)]["accuracy"])
            for mask_seed in MASK_SEEDS:
                r = float(index[common + (
                    "attnres", "final", "random_count_matched", str(mask_seed),
                    "blimp", "linguistics_term", term,
                )]["accuracy"])
                draws.append({
                    "training_seed": seed,
                    "mask_seed": mask_seed,
                    "term": term,
                    "unmasked_accuracy": u,
                    "old_masked_accuracy": o,
                    "random_masked_accuracy": r,
                    "baseline_accuracy": b,
                    "raw_old_cost_pp": u - o,
                    "random_deletion_cost_pp": u - r,
                    "old_specific_excess_cost_pp": r - o,
                    "attnres_relative_gain_pp": u - b,
                    "absolute_ability_pp": u - 50.0,
                })
    contrasts = []
    for seed in SEEDS:
        for term in terms:
            selected = [row for row in draws if row["training_seed"] == seed and row["term"] == term]
            contrasts.append({
                "training_seed": seed,
                "term": term,
                "random_draws": len(selected),
                "random_mean_accuracy": float(np.mean([row["random_masked_accuracy"] for row in selected])),
                "old_specific_excess_cost_pp": float(np.mean([row["old_specific_excess_cost_pp"] for row in selected])),
                "raw_old_cost_pp": selected[0]["raw_old_cost_pp"],
                "random_deletion_cost_pp": float(np.mean([row["random_deletion_cost_pp"] for row in selected])),
                "attnres_relative_gain_pp": selected[0]["attnres_relative_gain_pp"],
                "absolute_ability_pp": selected[0]["absolute_ability_pp"],
                "draw_aggregate_min_pp": "",
                "draw_aggregate_max_pp": "",
                "draw_aggregate_sd_pp": "",
            })
    for term in terms:
        selected = [row for row in contrasts if row["term"] == term]
        per_draw = [
            float(np.mean([
                row["old_specific_excess_cost_pp"] for row in draws
                if row["term"] == term and row["mask_seed"] == mask_seed
            ]))
            for mask_seed in MASK_SEEDS
        ]
        contrasts.append({
            "training_seed": "mean",
            "term": term,
            "random_draws": len(MASK_SEEDS),
            "random_mean_accuracy": float(np.mean([row["random_mean_accuracy"] for row in selected])),
            "old_specific_excess_cost_pp": float(np.mean([row["old_specific_excess_cost_pp"] for row in selected])),
            "raw_old_cost_pp": float(np.mean([row["raw_old_cost_pp"] for row in selected])),
            "random_deletion_cost_pp": float(np.mean([row["random_deletion_cost_pp"] for row in selected])),
            "attnres_relative_gain_pp": float(np.mean([row["attnres_relative_gain_pp"] for row in selected])),
            "absolute_ability_pp": float(np.mean([row["absolute_ability_pp"] for row in selected])),
            "draw_aggregate_min_pp": min(per_draw),
            "draw_aggregate_max_pp": max(per_draw),
            "draw_aggregate_sd_pp": float(np.std(per_draw, ddof=1)),
        })
    mean_rows = [row for row in contrasts if row["training_seed"] == "mean"]
    correlations = []
    for x_field in ("absolute_ability_pp", "attnres_relative_gain_pp"):
        x = np.array([float(row[x_field]) for row in mean_rows])
        y = np.array([float(row["old_specific_excess_cost_pp"]) for row in mean_rows])
        draw_r = []
        for mask_seed in MASK_SEEDS:
            draw_y = np.array([
                np.mean([
                    row["old_specific_excess_cost_pp"] for row in draws
                    if row["term"] == term and row["mask_seed"] == mask_seed
                ])
                for term in terms
            ])
            draw_r.append(float(np.corrcoef(x, draw_y)[0, 1]))
        correlations.append({
            "x_axis": x_field,
            "y_axis": "old_specific_excess_cost_pp",
            "n_terms": len(terms),
            "five_draw_mean_r": float(np.corrcoef(x, y)[0, 1]),
            "per_draw_r_min": min(draw_r),
            "per_draw_r_max": max(draw_r),
            **{f"r_seed_{mask_seed}": value for mask_seed, value in zip(MASK_SEEDS, draw_r, strict=True)},
        })
    return draws, contrasts, correlations


def fresh_100m_summary(behavior: list[dict], dev: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    plotting = {}
    for seed in SEEDS:
        plotting[seed] = {}
        for metric, spec in METRICS.items():
            series = paired_series(behavior, dev, "100m", seed, spec)
            plotting[seed][metric] = series
            delta = series["attnres_accuracy"] - series["baseline_accuracy"]
            for label, words, value in zip(series["labels"], series["words"], delta, strict=True):
                rows.append({
                    "row_type": "point", "seed": seed, "metric": metric,
                    "checkpoint_label": label, "words_seen": int(words), "delta": float(value),
                    "log_exposure_auc": "", "late_fresh_50_90m_mean_delta": "",
                })
            rows.append({
                "row_type": "aggregate", "seed": seed, "metric": metric,
                "checkpoint_label": "10M-90M", "words_seen": "", "delta": "",
                "log_exposure_auc": normalized_log_auc(series["words"], delta),
                "late_fresh_50_90m_mean_delta": float(np.mean(delta[series["words"] >= 50_000_000])),
            })
        series = plotting[seed]["overall_blimp"]
        nll_delta = series["baseline_nll"] - series["attnres_nll"]
        for label, words, value in zip(series["labels"], series["words"], nll_delta, strict=True):
            rows.append({
                "row_type": "point", "seed": seed, "metric": "dev_nll",
                "checkpoint_label": label, "words_seen": int(words), "delta": float(value),
                "log_exposure_auc": "", "late_fresh_50_90m_mean_delta": "",
            })
        rows.append({
            "row_type": "aggregate", "seed": seed, "metric": "dev_nll",
            "checkpoint_label": "10M-90M", "words_seen": "", "delta": "",
            "log_exposure_auc": normalized_log_auc(series["words"], nll_delta),
            "late_fresh_50_90m_mean_delta": float(np.mean(nll_delta[series["words"] >= 50_000_000])),
        })
    return rows, plotting


def render_figure_a(plotting: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8))
    for axis, metric in zip(axes[:2], ("overall_blimp", "filler_gap_dependency"), strict=True):
        for architecture in ("baseline", "attnres"):
            field = f"{architecture}_accuracy"
            for seed in SEEDS:
                series = plotting[metric][seed]
                axis.plot(series["words"], series[field], color=COLORS[architecture], alpha=.25, lw=1)
                pre = series["iters"] < 40
                axis.scatter(series["words"][pre], series[field][pre], facecolors="none",
                             edgecolors=COLORS[architecture], alpha=.45, s=22)
            mean_words = np.mean([plotting[metric][seed]["words"] for seed in SEEDS], axis=0)
            mean_accuracy = np.mean([plotting[metric][seed][field] for seed in SEEDS], axis=0)
            axis.plot(mean_words, mean_accuracy, color=COLORS[architecture], lw=2.4, label=architecture)
        axis.axvline(8e6, color="#9ca3af", ls="--", lw=1)
        axis.set_xscale("log")
        axis.set_title(metric.replace("_", " "), loc="left")
        axis.set_xlabel("words seen")
        axis.set_ylabel("accuracy (%)")
        axis.grid(alpha=.16)
    axes[0].legend(frameon=False)
    metric = "overall_blimp"
    for seed in SEEDS:
        series = plotting[metric][seed]
        axes[2].plot(series["words"], series["attnres_accuracy"] - series["baseline_accuracy"],
                     color=COLORS["attnres"], alpha=.35, marker="o", ms=2.5)
        for architecture in ("baseline", "attnres"):
            axes[3].plot(series[f"{architecture}_nll"], series[f"{architecture}_accuracy"],
                         color=COLORS[architecture], alpha=.4, marker="o", ms=2.5)
    axes[2].axhline(0, color="#9ca3af", lw=1)
    axes[2].set_xscale("log")
    axes[2].set_title("AttnRes − baseline", loc="left")
    axes[2].set_xlabel("words seen")
    axes[2].set_ylabel("overall BLiMP delta (pp)")
    axes[3].set_title("Behavior at matched loss", loc="left")
    axes[3].set_xlabel("fixed-dev NLL")
    axes[3].set_ylabel("overall BLiMP (%)")
    for axis in axes[2:]: axis.grid(alpha=.16)
    fig.suptitle("Dense 10M trajectory (19 points × 3 paired seeds; open markers are pre-warmup)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def render_figure_b(contrasts: list[dict], output: Path, raw_output: Path) -> None:
    rows = [row for row in contrasts if row["training_seed"] == "mean"]
    label_terms = {
        row["term"] for row in sorted(rows, key=lambda row: abs(float(row["attnres_relative_gain_pp"])), reverse=True)[:6]
    }
    y = np.array([float(row["old_specific_excess_cost_pp"]) for row in rows])
    low = y - np.array([float(row["draw_aggregate_min_pp"]) for row in rows])
    high = np.array([float(row["draw_aggregate_max_pp"]) for row in rows]) - y
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for axis, x_field, title in (
        (axes[0], "absolute_ability_pp", "(a) Absolute ability"),
        (axes[1], "attnres_relative_gain_pp", "(b) Relative gain"),
    ):
        x = np.array([float(row[x_field]) for row in rows])
        axis.errorbar(x, y, yerr=np.vstack([low, high]), fmt="o", color="#087e8b",
                      ecolor="#94a3b8", capsize=2.5, ms=5)
        for row, xv, yv in zip(rows, x, y, strict=True):
            if row["term"] in label_terms:
                axis.annotate(row["term"].replace("_", " "), (xv, yv), xytext=(4, 4),
                              textcoords="offset points", fontsize=7)
        axis.axhline(0, color="#b8b8b8", lw=1)
        axis.axvline(0, color="#b8b8b8", lw=1)
        axis.set_title(title, loc="left")
        axis.set_xlabel(x_field.replace("_pp", "").replace("_", " ") + " (pp)")
        axis.grid(alpha=.15)
    axes[0].set_ylabel("old-specific excess cost R̄ − O (pp)")
    fig.suptitle("Old routes are load-bearing, but excess cost does not align with AttnRes gains\n"
                 "n=13 BLiMP terms; whiskers are five random-control draws, not training seeds", fontsize=11)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.4, 4.8))
    x = np.array([float(row["absolute_ability_pp"]) for row in rows])
    raw = np.array([float(row["raw_old_cost_pp"]) for row in rows])
    axis.scatter(x, raw, color="#087e8b")
    axis.axhline(0, color="#b8b8b8", lw=1)
    axis.set_xlabel("absolute ability U − 50 (pp)")
    axis.set_ylabel("raw old-route cost U − O (pp)")
    axis.set_title("Raw collapse/headroom audit (n=13 BLiMP terms)", loc="left")
    axis.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(raw_output.with_suffix(".png"), dpi=220)
    fig.savefig(raw_output.with_suffix(".svg"))
    plt.close(fig)


def render_figure_d(plotting: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.9))
    for architecture in ("baseline", "attnres"):
        for seed in SEEDS:
            series = plotting[seed]["filler_gap_dependency"]
            axes[0].plot(series["words"], series[f"{architecture}_accuracy"],
                         color=COLORS[architecture], alpha=.28, lw=1)
        words = np.mean([plotting[seed]["filler_gap_dependency"]["words"] for seed in SEEDS], axis=0)
        accuracy = np.mean([
            plotting[seed]["filler_gap_dependency"][f"{architecture}_accuracy"] for seed in SEEDS
        ], axis=0)
        axes[0].plot(words, accuracy, color=COLORS[architecture], lw=2.4, label=architecture)
    for seed in SEEDS:
        series = plotting[seed]["filler_gap_dependency"]
        axes[1].plot(series["words"], series["attnres_accuracy"] - series["baseline_accuracy"],
                     marker="o", ms=3, alpha=.4, color=COLORS["attnres"])
        overall = plotting[seed]["overall_blimp"]
        axes[2].plot(overall["words"], overall["baseline_nll"] - overall["attnres_nll"],
                     marker="o", ms=3, alpha=.4, color="#7c3aed")
    axes[0].legend(frameon=False)
    axes[0].set_ylabel("filler-gap accuracy (%)")
    axes[1].set_ylabel("filler-gap AttnRes − baseline (pp)")
    axes[2].set_ylabel("dev NLL baseline − AttnRes")
    for axis, title in zip(axes, ("Fresh-data trajectory", "Paired behavioral delta", "Optimization context"), strict=True):
        axis.axhline(0, color="#b8b8b8", lw=1)
        axis.set_xscale("log")
        axis.set_xlabel("words seen")
        axis.set_title(title, loc="left")
        axis.grid(alpha=.15)
    fig.suptitle("100M first-epoch replication (9 points × 3 paired seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def dashboard_status(
    behavior: list[dict],
    dev: list[dict],
    output_dir: Path,
    figure_dir: Path,
    replay_grade: str,
    input_dir: Path | None = None,
) -> dict:
    a_blimp = {
        (row["run_name"], row["checkpoint_label"])
        for row in behavior if corpus_for(row) == "10m" and row["mask_mode"] == "none"
        and row["task"] == "blimp" and row["level"] == "overall"
    }
    a_dev = {(row["run_name"], row["checkpoint_label"]) for row in dev if corpus_for(row) == "10m"}
    b = {
        (row["seed"], row["mask_seed"])
        for row in behavior if row["mask_mode"] == "random_count_matched"
        and row["task"] == "blimp" and row["level"] == "overall"
    }
    c_blimp = {
        (row["run_name"], row["checkpoint_label"])
        for row in behavior if corpus_for(row) == "100m" and row["mask_mode"] == "none"
        and row["task"] == "blimp" and row["level"] == "overall"
    }
    c_dev = {(row["run_name"], row["checkpoint_label"]) for row in dev if corpus_for(row) == "100m"}
    required_figures = [
        "real_figA_dense_10m.png", "real_figA_dense_10m.svg",
        "real_figB_excess_cost.png", "real_figB_excess_cost.svg",
        "real_figB_raw_appendix.png", "real_figB_raw_appendix.svg",
        "real_figC_position_loss.png", "real_figC_position_loss.svg",
        "real_figD_100m_fresh.png", "real_figD_100m_fresh.svg",
    ]
    required_input_tables = [
        "diag_supp_checkpoint_inventory.csv", "diag_supp_behavior_long.csv",
        "diag_supp_dev_loss.csv",
    ]
    required_output_tables = [
        "diag_supp_trajectory_summary.csv", "diag_supp_masking_draws.csv",
        "diag_supp_masking_contrasts.csv",
        "diag_supp_100m_fresh_summary.csv", "diag_supp_replay_quality.csv",
    ]
    input_dir = input_dir or output_dir
    artifacts = sum((figure_dir / name).is_file() for name in required_figures)
    artifacts += sum((input_dir / name).is_file() for name in required_input_tables)
    artifacts += sum((output_dir / name).is_file() for name in required_output_tables)
    artifact_total = (
        len(required_figures) + len(required_input_tables) + len(required_output_tables)
    )
    children = [
        {"id": "diagnosis-supp-a-dense-10m", "label": "A — 10M dense trajectory",
         "blimp_progress": len(a_blimp), "blimp_total": 114,
         "dev_nll_progress": len(a_dev), "dev_nll_total": 114,
         "status": "complete" if len(a_blimp) == len(a_dev) == 114 else "running"},
        {"id": "diagnosis-supp-b-random-controls", "label": "B — five random controls",
         "blimp_progress": len(b), "blimp_total": 15,
         "status": "complete" if len(b) == 15 else "running"},
        {"id": "diagnosis-supp-c-100m-fresh", "label": "C — 100M fresh-data",
         "blimp_progress": len(c_blimp), "blimp_total": 54,
         "dev_nll_progress": len(c_dev), "dev_nll_total": 54,
         "replay_quality": replay_grade,
         "status": "complete" if len(c_blimp) == len(c_dev) == 54
         and replay_grade in {"green", "yellow", "red"} else "running"},
        {"id": "diagnosis-supp-d-figures", "label": "D — figures and summaries",
         "artifact_progress": artifacts, "artifact_total": artifact_total,
         "status": "complete" if artifacts == artifact_total else "running"},
    ]
    return {
        "id": "diagnosis-supp",
        "label": "诊断分析-补充",
        "status": "complete" if all(child["status"] == "complete" for child in children) else "running",
        "children": children,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--old-figure-dir", type=Path, required=True)
    parser.add_argument(
        "--replay-grade", choices=("pending", "green", "yellow", "red"), default="pending"
    )
    parser.add_argument("--replay-max-nll-difference", type=float)
    parser.add_argument("--replay-max-blimp-difference-pp", type=float)
    parser.add_argument("--replay-note", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    behavior = read_csv(args.input_dir / "diag_supp_behavior_long.csv")
    dev = read_csv(args.input_dir / "diag_supp_dev_loss.csv")
    if len({row["val_bin_sha256"] for row in dev if corpus_for(row) == "10m"}) != 1:
        raise RuntimeError("10M dev evaluations do not share one pinned val.bin")
    if len({row["val_bin_sha256"] for row in dev if corpus_for(row) == "100m"}) != 1:
        raise RuntimeError("100M dev evaluations do not share one pinned val.bin")
    trajectory_rows, plotting_a = trajectory_summaries(behavior, dev)
    draw_rows, contrast_rows, correlation_rows = masking_tables(behavior)
    fresh_rows, plotting_d = fresh_100m_summary(behavior, dev)
    fields = (
        "metric", "seed", "sensitivity", "n_points", "log_exposure_auc_pp",
        "largest_observed_label", "largest_observed_words", "largest_observed_delta_pp",
        "positive_window_start", "positive_window_end", "positive_window_points",
        "loss_matched_mean_delta_pp", "loss_matched_auc_pp", "n_loss_matched",
        "pava_applied", "final_outside_baseline_overlap",
    )
    write_csv(args.output_dir / "diag_supp_trajectory_summary.csv", trajectory_rows, fields)
    write_csv(args.output_dir / "diag_supp_masking_draws.csv", draw_rows, tuple(draw_rows[0]))
    write_csv(args.output_dir / "diag_supp_masking_contrasts.csv", contrast_rows, tuple(contrast_rows[0]))
    write_csv(args.output_dir / "diag_supp_masking_correlations.csv", correlation_rows, tuple(correlation_rows[0]))
    write_csv(args.output_dir / "diag_supp_100m_fresh_summary.csv", fresh_rows, tuple(fresh_rows[0]))
    write_csv(args.output_dir / "diag_supp_replay_quality.csv", [{
        "status": "complete" if args.replay_grade != "pending" else "pending",
        "grade": "" if args.replay_grade == "pending" else args.replay_grade,
        "max_nll_difference": (
            "" if args.replay_max_nll_difference is None else args.replay_max_nll_difference
        ),
        "max_blimp_difference_pp": (
            "" if args.replay_max_blimp_difference_pp is None
            else args.replay_max_blimp_difference_pp
        ),
        "note": args.replay_note or "populate after three-reference comparison",
    }], ("status", "grade", "max_nll_difference", "max_blimp_difference_pp", "note"))
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    render_figure_a(plotting_a, args.figure_dir / "real_figA_dense_10m")
    render_figure_b(
        contrast_rows,
        args.figure_dir / "real_figB_excess_cost",
        args.figure_dir / "real_figB_raw_appendix",
    )
    for suffix in (".png", ".svg"):
        shutil.copy2(
            args.old_figure_dir / f"real_figC_position_loss{suffix}",
            args.figure_dir / f"real_figC_position_loss{suffix}",
        )
    render_figure_d(plotting_d, args.figure_dir / "real_figD_100m_fresh")
    status = dashboard_status(
        behavior, dev, args.output_dir, args.figure_dir, args.replay_grade,
        input_dir=args.input_dir,
    )
    status_path = args.output_dir / "diag_supp_dashboard_status.json"
    if status_path.exists():
        raise FileExistsError(status_path)
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
