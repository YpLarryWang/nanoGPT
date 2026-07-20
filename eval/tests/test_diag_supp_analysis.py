from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from diag_supp_analysis import MASK_SEEDS, SEEDS, longest_positive_window, masking_tables  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
