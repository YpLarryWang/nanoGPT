from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from diag_parse_results import (  # noqa: E402
    parse_accuracy_report,
    run_metadata,
    source_host_for_plan,
    validate_blimp_predictions,
)


class ParseDiagnosisResultsTest(unittest.TestCase):
    def test_parses_named_sections_and_overall(self):
        text = """TEMPERATURE: 1.00

### UID ACCURACY
item_a: 51.25
item_b: 49.75

### LINGUISTICS_TERM ACCURACY
filler_gap_dependency: 68.50

### AVERAGE ACCURACY
56.50
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_temperature_report.txt"
            path.write_text(text, encoding="utf-8")
            rows = parse_accuracy_report(path)
        self.assertEqual(
            rows,
            [
                {"level": "uid", "term": "item_a", "accuracy": 51.25},
                {"level": "uid", "term": "item_b", "accuracy": 49.75},
                {
                    "level": "linguistics_term",
                    "term": "filler_gap_dependency",
                    "accuracy": 68.5,
                },
                {"level": "overall", "term": "overall", "accuracy": 56.5},
            ],
        )

    def test_run_metadata_uses_default_and_explicit_seeds(self):
        self.assertEqual(run_metadata("bl10m-d512L32-offdev"), ("baseline", 1337))
        self.assertEqual(
            run_metadata("bl10m-d512L32-attnres8-offdev-s1339"),
            ("attnres", 1339),
        )

    def test_rejects_incomplete_blimp_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.json"
            path.write_text('{"one_task": {"predictions": [{"id": "only_item"}]}}')
            with self.assertRaisesRegex(RuntimeError, "67 BLiMP task"):
                validate_blimp_predictions(path)

    def test_source_host_comes_from_series_root_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            host_root = Path(tmp) / "vast2_3090" / "series"
            plan = host_root / "run" / "plan.json"
            self.assertEqual(
                source_host_for_plan(plan, [host_root]),
                "vast2_3090",
            )


if __name__ == "__main__":
    unittest.main()
