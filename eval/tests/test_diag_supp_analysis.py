from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from diag_supp_analysis import (  # noqa: E402
    MASK_SEEDS,
    SEEDS,
    dashboard_status,
    longest_positive_window,
    masking_tables,
)


def behavior_row(seed, architecture, mode, term, accuracy, mask_seed=""):
    return {
        "run_name": f"bl10m-example-s{seed}",
        "corpus": "10m",
        "seed": str(seed),
        "architecture": architecture,
        "checkpoint_label": "final",
        "mask_mode": mode,
        "mask_seed": str(mask_seed),
        "task": "blimp",
        "level": "linguistics_term",
        "term": term,
        "accuracy": str(accuracy),
    }


class DiagnosisSupplementAnalysisTest(unittest.TestCase):
    def test_longest_positive_window_uses_observed_points(self):
        self.assertEqual(
            longest_positive_window(["1M", "2M", "3M", "4M", "5M"], [-1, 1, 2, 3, -1]),
            ("2M", "4M", 3),
        )

    def test_random_draws_average_within_training_seed_first(self):
        rows = []
        terms = [f"term_{index:02d}" for index in range(13)]
        for seed in SEEDS:
            for term_index, term in enumerate(terms):
                unmasked = 60 + term_index
                rows.extend([
                    behavior_row(seed, "attnres", "none", term, unmasked),
                    behavior_row(seed, "baseline", "none", term, unmasked - 1),
                    behavior_row(seed, "attnres", "old", term, unmasked - 4),
                ])
                for draw_index, mask_seed in enumerate(MASK_SEEDS):
                    rows.append(
                        behavior_row(
                            seed,
                            "attnres",
                            "random_count_matched",
                            term,
                            unmasked - draw_index,
                            mask_seed,
                        )
                    )
        draws, contrasts, correlations = masking_tables(rows)
        self.assertEqual(len(draws), 13 * 3 * 5)
        seed_row = next(
            row for row in contrasts
            if row["training_seed"] == 1337 and row["term"] == "term_00"
        )
        # Rbar = U - mean(0,1,2,3,4) = U - 2; O = U - 4; Rbar - O = 2.
        self.assertAlmostEqual(seed_row["old_specific_excess_cost_pp"], 2.0)
        self.assertEqual(len(correlations), 2)
        self.assertTrue(all(row["n_terms"] == 13 for row in correlations))

    def test_dashboard_d_uses_frozen_eighteen_artifact_contract(self):
        required_tables = (
            "diag_supp_checkpoint_inventory.csv",
            "diag_supp_behavior_long.csv",
            "diag_supp_dev_loss.csv",
            "diag_supp_trajectory_summary.csv",
            "diag_supp_masking_draws.csv",
            "diag_supp_masking_contrasts.csv",
            "diag_supp_100m_fresh_summary.csv",
            "diag_supp_replay_quality.csv",
        )
        required_figures = (
            "real_figA_dense_10m.png", "real_figA_dense_10m.svg",
            "real_figB_excess_cost.png", "real_figB_excess_cost.svg",
            "real_figB_raw_appendix.png", "real_figB_raw_appendix.svg",
            "real_figC_position_loss.png", "real_figC_position_loss.svg",
            "real_figD_100m_fresh.png", "real_figD_100m_fresh.svg",
        )
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp) / "tables"
            figure_dir = Path(temp) / "figures"
            output_dir.mkdir()
            figure_dir.mkdir()
            for name in required_tables:
                (output_dir / name).touch()
            for name in required_figures:
                (figure_dir / name).touch()

            status = dashboard_status([], [], output_dir, figure_dir, "yellow")

        child = next(
            row for row in status["children"] if row["id"] == "diagnosis-supp-d-figures"
        )
        self.assertEqual(child["artifact_progress"], 18)
        self.assertEqual(child["artifact_total"], 18)
        self.assertEqual(child["status"], "complete")


if __name__ == "__main__":
    unittest.main()
