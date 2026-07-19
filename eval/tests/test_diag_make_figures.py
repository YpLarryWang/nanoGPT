from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from diag_make_figures import (  # noqa: E402
    CHECKPOINT_LABELS,
    final_position_curves,
    interpolate_on_nonincreasing_x,
    masking_contrasts,
    normalized_log_auc,
    pava_nonincreasing,
    render_figure_a,
    render_figure_b,
    render_figure_c,
    selected_blimp_terms,
    trajectory_summaries,
    weighted_position_bins,
)


class DiagnosisAnalysisMathTest(unittest.TestCase):
    def test_normalized_log_auc_preserves_constant_delta_units(self):
        words = [1e6, 5e6, 10e6, 50e6]
        self.assertAlmostEqual(normalized_log_auc(words, [2.5] * 4), 2.5, places=12)

    def test_pava_pools_only_nonmonotonic_loss_inversion(self):
        fitted = pava_nonincreasing([4.50, 4.30, 4.24, 4.25, 4.18])
        np.testing.assert_allclose(fitted, [4.50, 4.30, 4.245, 4.245, 4.18])

    def test_loss_interpolation_averages_accuracy_on_exact_plateau(self):
        nll = [4.50, 4.30, 4.245, 4.245, 4.18]
        accuracy = [50, 55, 60, 64, 70]
        self.assertEqual(interpolate_on_nonincreasing_x(4.245, nll, accuracy), 62.0)
        self.assertAlmostEqual(
            interpolate_on_nonincreasing_x(4.34, nll, accuracy),
            54.0,
            places=12,
        )
        self.assertIsNone(interpolate_on_nonincreasing_x(4.6, nll, accuracy))

    def test_unequal_position_bins_are_token_weighted(self):
        contexts = np.arange(1, 9)
        deltas = np.array([1, 1, 3, 3, 3, 3, 3, 3], dtype=float)
        counts = np.ones(8, dtype=int)
        bins = weighted_position_bins(contexts, deltas, counts, bins=((1, 2), (3, 8)))
        self.assertEqual([row["delta"] for row in bins], [1.0, 3.0])
        token_weighted = sum(row["token_count"] * row["delta"] for row in bins) / sum(
            row["token_count"] for row in bins
        )
        self.assertEqual(token_weighted, 2.5)
        self.assertNotEqual(token_weighted, np.mean([row["delta"] for row in bins]))

    def test_end_to_end_summary_contracts_on_synthetic_three_seed_grid(self):
        terms = ["filler_gap_dependency", *(f"term_{index}" for index in range(12))]
        behavior = []
        dev = []
        positions = []
        for seed in (1337, 1338, 1339):
            for architecture in ("baseline", "attnres"):
                for index, label in enumerate(CHECKPOINT_LABELS):
                    words = [1, 5, 10, 20, 50, 90][index] * 1_000_000
                    nll = 4.6 - 0.08 * index - (0.02 if architecture == "attnres" else 0)
                    dev.append(
                        {
                            "seed": str(seed),
                            "architecture": architecture,
                            "checkpoint_label": label,
                            "words_seen": str(words),
                            "mean_nll": str(nll),
                        }
                    )
                    for level, term, accuracy in (
                        ("linguistics_term", "filler_gap_dependency", 50 + index),
                        ("overall", "overall", 60 + index),
                    ):
                        behavior.append(
                            {
                                "seed": str(seed),
                                "architecture": architecture,
                                "checkpoint_label": label,
                                "mask_mode": "none",
                                "task": "blimp",
                                "level": level,
                                "term": term,
                                "accuracy": str(accuracy + (1 if architecture == "attnres" else 0)),
                            }
                        )
                # Add the other final BLiMP terms; filler and overall already exist.
                for term_index, term in enumerate(terms[1:], start=1):
                    behavior.append(
                        {
                            "seed": str(seed),
                            "architecture": architecture,
                            "checkpoint_label": "final",
                            "mask_mode": "none",
                            "task": "blimp",
                            "level": "linguistics_term",
                            "term": term,
                            "accuracy": str(
                                55
                                + term_index
                                + (term_index % 3 - 1 if architecture == "attnres" else 0)
                            ),
                        }
                    )
            for level, term in [
                ("linguistics_term", term) for term in terms
            ] + [("overall", "overall")]:
                unmasked = next(
                    float(row["accuracy"])
                    for row in behavior
                    if row["seed"] == str(seed)
                    and row["architecture"] == "attnres"
                    and row["checkpoint_label"] == "final"
                    and row["mask_mode"] == "none"
                    and row["level"] == level
                    and row["term"] == term
                )
                for mode, cost in (("old", 2.0), ("embed", 1.0), ("random_count_matched", 0.5)):
                    behavior.append(
                        {
                            "seed": str(seed),
                            "architecture": "attnres",
                            "checkpoint_label": "final",
                            "mask_mode": mode,
                            "task": "blimp",
                            "level": level,
                            "term": term,
                            "accuracy": str(unmasked - cost),
                        }
                    )
            for loss_index in range(512):
                for architecture, mean_nll in (("baseline", 4.0), ("attnres", 3.98)):
                    positions.append(
                        {
                            "seed": str(seed),
                            "architecture": architecture,
                            "checkpoint_label": "final",
                            "loss_index": str(loss_index),
                            "context_length": str(loss_index + 1),
                            "token_count": "3",
                            "mean_nll": str(mean_nll),
                        }
                    )
            for row in dev:
                if row["seed"] == str(seed) and row["checkpoint_label"] == "final":
                    row["mean_nll"] = "4.0" if row["architecture"] == "baseline" else "3.98"

        summaries, plotting = trajectory_summaries(behavior, dev)
        self.assertEqual(len(summaries), 6)
        contrasts = masking_contrasts(behavior)
        self.assertEqual(len(contrasts), 14 * 4)
        selected = selected_blimp_terms(behavior)
        self.assertEqual(len(selected), 9)
        curves = final_position_curves(positions, dev)
        self.assertEqual(set(curves), {1337, 1338, 1339})
        for curve in curves.values():
            self.assertAlmostEqual(curve["overall"], 0.02, places=12)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            render_figure_a(plotting, output / "figure_a")
            render_figure_b(contrasts, selected, output / "figure_b")
            render_figure_c(curves, output / "figure_c")
            for name in ("figure_a", "figure_b", "figure_c"):
                self.assertGreater((output / f"{name}.png").stat().st_size, 0)
                self.assertGreater((output / f"{name}.svg").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
