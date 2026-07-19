from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from diag_dev_series import CHECKPOINT_LABELS, diagnosis_plan  # noqa: E402


def entry(path, role, iteration, words, labels):
    return {
        "path": path,
        "role": role,
        "iter_num": iteration,
        "tokens_seen": words * 2,
        "words_seen": words,
        "labels": labels,
    }


class DiagnosisPlanTest(unittest.TestCase):
    def manifest(self):
        checkpoints = []
        for iteration, label in enumerate(CHECKPOINT_LABELS[:-1], start=1):
            words = int(label[:-1]) * 1_000_000
            checkpoints.append(
                entry(
                    f"ckpt_{iteration:06d}.pt",
                    "milestone",
                    iteration,
                    words,
                    [
                        {
                            "series": "words",
                            "name": f"words_{label}",
                            "revision": f"chck_{label}",
                        }
                    ],
                )
            )
        # Three physical files share the final iter and labels. Role must decide.
        final_labels = [{"series": "words", "name": "words_final", "revision": "chck_90M"}]
        checkpoints.extend(
            [
                entry("ckpt_000471.pt", "milestone", 471, 90_000_000, final_labels),
                entry("ckpt_best.pt", "best", 471, 90_000_000, final_labels),
                entry("ckpt_final.pt", "final", 471, 90_000_000, final_labels),
            ]
        )
        return {
            "run_name": "example-s1337",
            "roles": {"best": "ckpt_best.pt", "final": "ckpt_final.pt"},
            "checkpoints": checkpoints,
        }

    def test_selects_exact_six_points_and_physical_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = diagnosis_plan(Path(tmp), self.manifest())
        self.assertEqual([item["checkpoint_label"] for item in plan], list(CHECKPOINT_LABELS))
        self.assertEqual(plan[-1]["checkpoint_filename"], "ckpt_final.pt")
        self.assertEqual(plan[-1]["checkpoint_role"], "final")
        self.assertEqual(plan[0]["blimp_model_name"], "example-s1337--diag-words-1M")
        self.assertEqual(plan[-1]["blimp_model_name"], "example-s1337")

    def test_missing_required_word_point_is_fatal(self):
        manifest = self.manifest()
        manifest["checkpoints"] = [
            item
            for item in manifest["checkpoints"]
            if not any(label.get("name") == "words_20M" for label in item["labels"])
        ]
        with self.assertRaisesRegex(ValueError, "20M"):
            diagnosis_plan(Path("/unused"), manifest)


if __name__ == "__main__":
    unittest.main()
