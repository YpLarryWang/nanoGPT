import tempfile
import unittest
import csv
from pathlib import Path

from eval.sync_eval_results import (
    collect_glue,
    discover_revisions,
    parse_average_accuracy,
    parse_reading_report,
)
from eval.validate_results import ERRORS, rows


class SyncEvalResultsTest(unittest.TestCase):
    def test_glue_uses_declared_metric_key(self):
        with tempfile.TemporaryDirectory() as temp:
            strict = Path(temp)
            base = strict / "results" / "model" / "main" / "finetune"
            metrics = {
                "boolq": (0.71, 0.41), "multirc": (0.61, 0.39),
                "rte": (0.59, 0.22), "wsc": (0.63, 0.24),
                "mrpc": (0.70, 0.82), "qqp": (0.76, 0.73),
                "mnli": (0.62, None),
            }
            for task, (accuracy, f1) in metrics.items():
                path = base / task / "results.txt"
                path.parent.mkdir(parents=True)
                text = f"accuracy: {accuracy}\n"
                if f1 is not None:
                    text += f"f1: {f1}\n"
                path.write_text(text)
            result = collect_glue(strict, "model", "main")
            self.assertEqual(result["multirc_acc"], "61.00")
            self.assertEqual(result["mrpc_f1"], "82.00")
            self.assertEqual(result["qqp_f1"], "73.00")

    def test_zero_shot_report_requires_labelled_average(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.txt"
            path.write_text("### UID ACCURACY\na: 99.00\n\n### AVERAGE ACCURACY\n57.25\n")
            self.assertEqual(parse_average_accuracy(path), 57.25)

    def test_reading_uses_explicit_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.txt"
            path.write_text("SELF-PACED READING SCORE: 4.47\nEYE TRACKING SCORE: 8.70\n")
            self.assertEqual(parse_reading_report(path), (8.70, 4.47))

    def test_fast_revision_discovery_excludes_full_only_main(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "results" / "model"
            full = root / "main" / "zero_shot" / "causal" / "blimp" / "blimp_filtered"
            fast = root / "chck_10M" / "zero_shot" / "causal" / "blimp" / "blimp_fast"
            full.mkdir(parents=True)
            fast.mkdir(parents=True)
            (full / "best_temperature_report.txt").write_text("full")
            (fast / "best_temperature_report.txt").write_text("fast")
            self.assertEqual(discover_revisions(Path(temp), "model", "causal"), ["chck_10M"])

    def test_fast_csv_allows_multiple_revisions_for_one_model(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fast_zero_shot.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("model", "revision", "backend"))
                writer.writeheader()
                writer.writerows([
                    {"model": "model", "revision": "chck_1M", "backend": "causal"},
                    {"model": "model", "revision": "chck_10M", "backend": "causal"},
                ])
            ERRORS.clear()
            parsed = rows(path, key_fields=("model", "revision", "backend"))
            self.assertEqual(len(parsed), 2)
            self.assertEqual(ERRORS, [])


if __name__ == "__main__":
    unittest.main()
