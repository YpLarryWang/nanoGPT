#!/usr/bin/env python3
"""Compute frozen diagnosis summaries and render the three paper figures."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CHECKPOINT_LABELS = ("1M", "5M", "10M", "20M", "50M", "final")
SEEDS = (1337, 1338, 1339)
METRICS = {
    "filler_gap_dependency": ("blimp", "linguistics_term", "filler_gap_dependency"),
    "overall_blimp": ("blimp", "overall", "overall"),
}
MASK_MODES = ("old", "embed", "random_count_matched")
POSITION_BINS = ((1, 4), (5, 16), (17, 64), (65, 256), (257, 512))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_log_auc(words, deltas) -> float:
    words = np.asarray(words, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    if len(words) < 2 or len(words) != len(deltas):
        raise ValueError("AUC requires equally sized arrays with at least two points")
    if np.any(words <= 0) or np.any(np.diff(words) <= 0):
        raise ValueError("AUC words must be positive and strictly increasing")
    log_words = np.log(words)
    return float(np.trapz(deltas, x=log_words) / (log_words[-1] - log_words[0]))


def pava_nonincreasing(values) -> np.ndarray:
    """Equal-weight PAVA projection onto a non-increasing sequence."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("PAVA requires a finite one-dimensional sequence")
    blocks = []
    for index, value in enumerate(values):
        blocks.append([index, index + 1, 1.0, float(value)])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            left_mean = left[3] / left[2]
            right_mean = right[3] / right[2]
            if left_mean >= right_mean:
                break
            blocks[-2:] = [[left[0], right[1], left[2] + right[2], left[3] + right[3]]]
    fitted = np.empty_like(values)
    for start, end, weight, total in blocks:
        fitted[start:end] = total / weight
    return fitted


def interpolate_on_nonincreasing_x(target: float, x, y) -> float | None:
    """Interpolate y at target x; average y only when target hits a PAVA plateau."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y) or len(x) == 0 or np.any(np.diff(x) > 1e-12):
        raise ValueError("x must be a non-increasing sequence matching y")
    if target > x[0] + 1e-12 or target < x[-1] - 1e-12:
        return None
    exact = np.isclose(x, target, rtol=0.0, atol=1e-12)
    if exact.any():
        return float(y[exact].mean())
    for index in range(len(x) - 1):
        high, low = x[index], x[index + 1]
        if high == low:
            continue
        if high > target > low:
            fraction = (high - target) / (high - low)
            return float(y[index] + fraction * (y[index + 1] - y[index]))
    raise RuntimeError(f"failed to interpolate in-range target {target}")


def weighted_position_bins(context_lengths, deltas, token_counts, bins=POSITION_BINS):
    contexts = np.asarray(context_lengths, dtype=np.int64)
    deltas = np.asarray(deltas, dtype=np.float64)
    counts = np.asarray(token_counts, dtype=np.int64)
    if not (len(contexts) == len(deltas) == len(counts)):
        raise ValueError("position arrays must have equal lengths")
    if np.any(counts <= 0):
        raise ValueError("position token counts must be positive")
    output = []
    for start, end in bins:
        selected = (contexts >= start) & (contexts <= end)
        if not selected.any():
            raise ValueError(f"empty position bin {start}-{end}")
        weight = int(counts[selected].sum())
        delta = float(np.dot(counts[selected], deltas[selected]) / weight)
        output.append(
            {
                "start": start,
                "end": end,
                "x": math.sqrt(start * end),
                "token_count": weight,
                "delta": delta,
            }
        )
    return output


def write_csv_exclusive(path: Path, rows: list[dict], fieldnames: tuple[str, ...]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def unique_index(rows: list[dict], keys: tuple[str, ...], label: str) -> dict:
    output = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        if key in output:
            raise ValueError(f"duplicate {label} key: {key}")
        output[key] = row
    return output


def behavior_index(rows: list[dict]) -> dict:
    return unique_index(
        rows,
        ("seed", "architecture", "checkpoint_label", "mask_mode", "task", "level", "term"),
        "behavior",
    )


def dev_index(rows: list[dict]) -> dict:
    return unique_index(rows, ("seed", "architecture", "checkpoint_label"), "dev")


def metric_accuracy(index: dict, seed: int, architecture: str, label: str, spec) -> float:
    task, level, term = spec
    key = (str(seed), architecture, label, "none", task, level, term)
    if key not in index:
        raise KeyError(f"missing behavior row: {key}")
    return float(index[key]["accuracy"])


def trajectory_summaries(behavior_rows: list[dict], dev_rows: list[dict]):
    bidx = behavior_index(behavior_rows)
    didx = dev_index(dev_rows)
    summaries = []
    plotting = {}
    for metric, spec in METRICS.items():
        plotting[metric] = {}
        for seed in SEEDS:
            baseline_acc = np.array([
                metric_accuracy(bidx, seed, "baseline", label, spec)
                for label in CHECKPOINT_LABELS
            ])
            attnres_acc = np.array([
                metric_accuracy(bidx, seed, "attnres", label, spec)
                for label in CHECKPOINT_LABELS
            ])
            baseline_dev = [didx[(str(seed), "baseline", label)] for label in CHECKPOINT_LABELS]
            attnres_dev = [didx[(str(seed), "attnres", label)] for label in CHECKPOINT_LABELS]
            baseline_words = np.array([float(row["words_seen"]) for row in baseline_dev])
            attnres_words = np.array([float(row["words_seen"]) for row in attnres_dev])
            if not np.array_equal(baseline_words, attnres_words):
                raise RuntimeError(f"paired words differ for seed {seed}")
            baseline_nll = np.array([float(row["mean_nll"]) for row in baseline_dev])
            attnres_nll = np.array([float(row["mean_nll"]) for row in attnres_dev])
            iso_nll = pava_nonincreasing(baseline_nll)
            matched_deltas = []
            matched_words = []
            for words, target_nll, accuracy in zip(
                attnres_words, attnres_nll, attnres_acc, strict=True
            ):
                matched_baseline = interpolate_on_nonincreasing_x(
                    float(target_nll), iso_nll, baseline_acc
                )
                if matched_baseline is not None:
                    matched_words.append(words)
                    matched_deltas.append(float(accuracy - matched_baseline))
            matched_auc = (
                normalized_log_auc(matched_words, matched_deltas)
                if len(matched_words) >= 2
                else math.nan
            )
            deltas = attnres_acc - baseline_acc
            summaries.append(
                {
                    "metric": metric,
                    "seed": seed,
                    "log_exposure_auc_pp": normalized_log_auc(attnres_words, deltas),
                    "loss_matched_mean_delta_pp": (
                        float(np.mean(matched_deltas)) if matched_deltas else math.nan
                    ),
                    "loss_matched_auc_pp": matched_auc,
                    "n_loss_matched": len(matched_deltas),
                    "pava_applied": int(not np.array_equal(iso_nll, baseline_nll)),
                }
            )
            plotting[metric][seed] = {
                "words": attnres_words,
                "baseline_accuracy": baseline_acc,
                "attnres_accuracy": attnres_acc,
                "baseline_nll": baseline_nll,
                "baseline_nll_isotonic": iso_nll,
                "attnres_nll": attnres_nll,
            }
    return summaries, plotting


def masking_contrasts(behavior_rows: list[dict]) -> list[dict]:
    index = behavior_index(behavior_rows)
    individual = []
    attn_keys = [
        key
        for key in index
        if key[1] == "attnres" and key[2] == "final" and key[3] == "none"
        and key[4] in {"blimp", "comps", "entity_tracking"}
    ]
    for key in sorted(attn_keys):
        seed, _, label, _, task, level, term = key
        baseline_key = (seed, "baseline", label, "none", task, level, term)
        if baseline_key not in index:
            raise KeyError(f"missing baseline final behavior: {baseline_key}")
        unmasked = float(index[key]["accuracy"])
        costs = {}
        for mode in MASK_MODES:
            masked_key = (seed, "attnres", label, mode, task, level, term)
            if masked_key not in index:
                raise KeyError(f"missing masked behavior: {masked_key}")
            costs[mode] = unmasked - float(index[masked_key]["accuracy"])
        individual.append(
            {
                "seed": seed,
                "task": task,
                "level": level,
                "term": term,
                "attnres_gain_pp": unmasked - float(index[baseline_key]["accuracy"]),
                "cost_old_pp": costs["old"],
                "cost_embed_pp": costs["embed"],
                "cost_random_count_matched_pp": costs["random_count_matched"],
                "old_excess_vs_random_pp": costs["old"] - costs["random_count_matched"],
            }
        )
    grouped = {}
    for row in individual:
        grouped.setdefault((row["task"], row["level"], row["term"]), []).append(row)
    means = []
    value_fields = (
        "attnres_gain_pp", "cost_old_pp", "cost_embed_pp",
        "cost_random_count_matched_pp", "old_excess_vs_random_pp",
    )
    for (task, level, term), rows in sorted(grouped.items()):
        if {int(row["seed"]) for row in rows} != set(SEEDS):
            raise RuntimeError(f"masking contrast lacks all three seeds: {(task, level, term)}")
        means.append(
            {
                "seed": "mean",
                "task": task,
                "level": level,
                "term": term,
                **{
                    field: float(np.mean([float(row[field]) for row in rows]))
                    for field in value_fields
                },
            }
        )
    return individual + means


def selected_blimp_terms(behavior_rows: list[dict]) -> set[str]:
    """Freeze 3 positive/3 negative/3 near-zero labels from unmasked data only."""
    index = behavior_index(behavior_rows)
    terms = sorted(
        {
            key[-1]
            for key in index
            if key[1] == "attnres" and key[2] == "final" and key[3] == "none"
            and key[4] == "blimp" and key[5] == "linguistics_term"
        }
    )
    if len(terms) != 13:
        raise RuntimeError(f"expected 13 unmasked BLiMP terms, got {len(terms)}")
    deltas = []
    for term in terms:
        seed_deltas = []
        for seed in SEEDS:
            attn_key = (str(seed), "attnres", "final", "none", "blimp", "linguistics_term", term)
            base_key = (str(seed), "baseline", "final", "none", "blimp", "linguistics_term", term)
            seed_deltas.append(
                float(index[attn_key]["accuracy"]) - float(index[base_key]["accuracy"])
            )
        deltas.append({"term": term, "delta": float(np.mean(seed_deltas))})
    by_delta = sorted(deltas, key=lambda row: row["delta"])
    selected = {row["term"] for row in by_delta[:3] + by_delta[-3:]}
    for row in sorted(deltas, key=lambda row: abs(row["delta"])):
        selected.add(row["term"])
        if len(selected) == 9:
            break
    return selected


def final_position_curves(position_rows: list[dict], dev_rows: list[dict]):
    pidx = unique_index(
        position_rows,
        ("seed", "architecture", "checkpoint_label", "loss_index"),
        "position",
    )
    didx = dev_index(dev_rows)
    curves = {}
    for seed in SEEDS:
        contexts = []
        deltas = []
        counts = []
        for loss_index in range(512):
            baseline = pidx[(str(seed), "baseline", "final", str(loss_index))]
            attnres = pidx[(str(seed), "attnres", "final", str(loss_index))]
            baseline_count = int(baseline["token_count"])
            attnres_count = int(attnres["token_count"])
            if baseline_count != attnres_count:
                raise RuntimeError(f"paired position counts differ: seed={seed} index={loss_index}")
            contexts.append(int(baseline["context_length"]))
            counts.append(baseline_count)
            deltas.append(float(baseline["mean_nll"]) - float(attnres["mean_nll"]))
        contexts = np.asarray(contexts)
        deltas = np.asarray(deltas)
        counts = np.asarray(counts)
        position_overall = float(np.dot(counts, deltas) / counts.sum())
        headline = (
            float(didx[(str(seed), "baseline", "final")]["mean_nll"])
            - float(didx[(str(seed), "attnres", "final")]["mean_nll"])
        )
        if not math.isclose(position_overall, headline, rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError(
                f"position/headline delta mismatch for seed {seed}: "
                f"{position_overall} != {headline}"
            )
        curves[seed] = {
            "bins": weighted_position_bins(contexts, deltas, counts),
            "overall": headline,
        }
    return curves


def render_figure_a(plotting: dict, output: Path) -> None:
    colors = {"baseline": "#6b7280", "attnres": "#087e8b"}
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7))
    for axis, metric, title in zip(
        axes[:2], METRICS, ("Filler-gap trajectory", "Overall BLiMP trajectory"), strict=True
    ):
        mean_words = np.mean([plotting[metric][seed]["words"] for seed in SEEDS], axis=0)
        for architecture in ("baseline", "attnres"):
            field = f"{architecture}_accuracy"
            for seed in SEEDS:
                row = plotting[metric][seed]
                axis.plot(row["words"], row[field], color=colors[architecture], alpha=0.28, lw=1)
            mean_accuracy = np.mean([plotting[metric][seed][field] for seed in SEEDS], axis=0)
            axis.plot(
                mean_words,
                mean_accuracy,
                color=colors[architecture],
                lw=2.4,
                label=architecture,
            )
        axis.set_xscale("log")
        axis.set_xlabel("words seen")
        axis.set_ylabel("accuracy (%)")
        axis.set_title(title, loc="left")
        axis.grid(alpha=0.18)
    axes[0].legend(frameon=False)
    phase = axes[2]
    metric = "filler_gap_dependency"
    for architecture in ("baseline", "attnres"):
        for seed in SEEDS:
            row = plotting[metric][seed]
            phase.plot(
                row[f"{architecture}_nll"], row[f"{architecture}_accuracy"],
                color=colors[architecture], alpha=0.6, marker="o", ms=3, lw=1.2,
            )
    phase.set_xlabel("fixed-dev NLL")
    phase.set_ylabel("filler-gap accuracy (%)")
    phase.set_title("Behavior at matched loss", loc="left")
    phase.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def render_figure_b(mask_rows: list[dict], selected: set[str], output: Path) -> None:
    terms = [
        row for row in mask_rows
        if row["seed"] == "mean" and row["task"] == "blimp"
        and row["level"] == "linguistics_term"
    ]
    fig, axis = plt.subplots(figsize=(6.4, 5.1))
    x = np.array([float(row["attnres_gain_pp"]) for row in terms])
    old = np.array([float(row["cost_old_pp"]) for row in terms])
    random_cost = np.array([float(row["cost_random_count_matched_pp"]) for row in terms])
    axis.axhline(0, color="#b8b8b8", lw=1)
    axis.axvline(0, color="#b8b8b8", lw=1)
    axis.scatter(x, old, color="#087e8b", s=42, label="mask old (age ≥ 2)")
    axis.scatter(x, random_cost, color="#d97706", marker="x", s=42, label="count-matched random")
    for row, x_value, y_value in zip(terms, x, old, strict=True):
        if row["term"] in selected:
            axis.annotate(row["term"].replace("_", " "), (x_value, y_value), xytext=(4, 4),
                          textcoords="offset points", fontsize=7)
    axis.set_xlabel("unmasked AttnRes gain over baseline (pp)")
    axis.set_ylabel("masking cost: unmasked − masked (pp)")
    axis.set_title("Do gains depend on older AttnRes routes?", loc="left")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def render_figure_c(curves: dict, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    seed_values = []
    for seed in SEEDS:
        bins = curves[seed]["bins"]
        x = [row["x"] for row in bins]
        y = [row["delta"] for row in bins]
        seed_values.append(y)
        axis.plot(x, y, color="#087e8b", alpha=0.32, lw=1.1, marker="o", ms=3)
    mean_y = np.mean(seed_values, axis=0)
    axis.plot(x, mean_y, color="#087e8b", lw=2.5, marker="o", label="mean across seed pairs")
    overall = float(np.mean([curves[seed]["overall"] for seed in SEEDS]))
    axis.axhline(overall, color="#6b7280", ls="--", lw=1.2,
                 label=f"token-weighted overall = {overall:.3f}")
    axis.axhline(0, color="#b8b8b8", lw=1)
    axis.set_xscale("log", base=2)
    axis.set_xticks(x, [f"{start}–{end}" for start, end in POSITION_BINS])
    axis.set_xlabel("context length bin")
    axis.set_ylabel("NLL baseline − AttnRes (nats)")
    axis.set_title("Where does the fixed-dev loss gain occur?", loc="left")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    behavior_rows = read_csv(input_dir / "diag_behavior_long.csv")
    dev_rows = read_csv(input_dir / "diag_dev_loss.csv")
    position_rows = read_csv(input_dir / "diag_position_nll.csv")
    if len({row["val_bin_sha256"] for row in dev_rows}) != 1:
        raise RuntimeError("all paired dev evaluations must use the same pinned val.bin")

    summaries, plotting = trajectory_summaries(behavior_rows, dev_rows)
    # Freeze labels from unmasked deltas before constructing or inspecting any
    # masking-cost contrast.
    selected_terms = selected_blimp_terms(behavior_rows)
    mask_rows = masking_contrasts(behavior_rows)
    for row in mask_rows:
        row["selected_main"] = int(
            row["task"] == "blimp"
            and row["level"] == "linguistics_term"
            and row["term"] in selected_terms
        )
    position_curves = final_position_curves(position_rows, dev_rows)
    write_csv_exclusive(output_dir / "diag_trajectory_summary.csv", summaries, (
        "metric", "seed", "log_exposure_auc_pp", "loss_matched_mean_delta_pp",
        "loss_matched_auc_pp", "n_loss_matched", "pava_applied",
    ))
    write_csv_exclusive(output_dir / "diag_masking_contrasts.csv", mask_rows, (
        "seed", "task", "level", "term", "attnres_gain_pp", "cost_old_pp",
        "cost_embed_pp", "cost_random_count_matched_pp", "old_excess_vs_random_pp",
        "selected_main",
    ))
    figure_dir.mkdir(parents=True, exist_ok=True)
    render_figure_a(plotting, figure_dir / "real_figA_dev_trajectories")
    render_figure_b(mask_rows, selected_terms, figure_dir / "real_figB_masking_cost")
    render_figure_c(position_curves, figure_dir / "real_figC_position_loss")
    print(
        f"wrote summaries={len(summaries)} masking_rows={len(mask_rows)} "
        f"figures={figure_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
