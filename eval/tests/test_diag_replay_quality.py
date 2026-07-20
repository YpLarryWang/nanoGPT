from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from diag_replay_quality import LABELS, compare  # noqa: E402


class ReplayQualityTest(unittest.TestCase):
    def test_green_grade_from_three_reference_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            replay = root / "replay"
            eval_root = root / "eval"
            for series, prefix, nll_offset, accuracy_offset in (
                (reference, "original", 0.0, 0.0),
                (replay, "replay", 0.005, 0.4),
            ):
                plan = []
                for index, label in enumerate(LABELS, start=1):
                    model_name = f"{prefix}-{label}"
                    plan.append({
                        "checkpoint_label": label,
                        "checkpoint_path": str(root / f"{prefix}-{label}.pt"),
                        "blimp_model_name": model_name,
                        "iter_num": index,
                        "words_seen": index * 10_000_000,
                    })
                    point = series / label
                    point.mkdir(parents=True)
                    (point / "dev_loss.json").write_text(
                        json.dumps({"mean_nll": 4.0 + nll_offset}), encoding="utf-8"
                    )
                    report = (
                        eval_root / "results" / model_name / "main" / "zero_shot" / "causal"
                        / "blimp" / "blimp_filtered" / "best_temperature_report.txt"
                    )
                    report.parent.mkdir(parents=True)
                    report.write_text(
                        "### LINGUISTICS_TERM ACCURACY\n"
                        f"filler_gap_dependency: {60 + accuracy_offset}\n\n"
                        "### AVERAGE ACCURACY\n"
                        f"{70 + accuracy_offset}\n",
                        encoding="utf-8",
                    )
                series.mkdir(exist_ok=True)
                (series / "plan.json").write_text(
                    json.dumps({"checkpoints": plan}), encoding="utf-8"
                )
            rows, summary = compare(reference, replay, eval_root, skip_tensors=True)
            self.assertEqual(len(rows), 3)
            self.assertEqual(summary["grade"], "green")
            self.assertAlmostEqual(summary["max_nll_difference"], 0.005)
            self.assertAlmostEqual(summary["max_overall_blimp_difference_pp"], 0.4)


if __name__ == "__main__":
    unittest.main()
