import tempfile
import unittest
from pathlib import Path

from eval.run_local_aoa import (
    local_step_plan,
    split_complete_resume_results,
    unfinished_steps,
    validate_complete_results,
)


class LocalStepPlanTest(unittest.TestCase):
    def test_disambiguates_duplicate_revision_without_dropping_points(self):
        manifest = {
            "checkpoints": [
                {
                    "path": "ckpt_000469.pt",
                    "role": "milestone",
                    "iter_num": 469,
                    "labels": [
                        {
                            "series": "words",
                            "revision": "chck_90M",
                            "target": 90_000_000,
                            "actual": 89_960_402,
                        }
                    ],
                },
                {
                    "path": "ckpt_000471.pt",
                    "role": "milestone",
                    "iter_num": 471,
                    "labels": [
                        {
                            "series": "words",
                            "revision": "chck_90M",
                            "target": 90_343_884,
                            "actual": 90_343_884,
                        }
                    ],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            plan = local_step_plan(Path(tmp), manifest, "words")

        self.assertEqual([item["step"] for item in plan], [90_000_000, 90_343_884])
        self.assertEqual([item["revision"] for item in plan], ["chck_90M", "chck_90M"])
        self.assertEqual(
            [item["cache_revision"] for item in plan],
            [
                "chck_90M-t90000000-i000469",
                "chck_90M-t90343884-i000471",
            ],
        )

    def test_resume_keeps_only_complete_finite_steps(self):
        plan = [{"step": 1}, {"step": 2}, {"step": 3}]

        def row(step, word, context_id, surprisal):
            return {
                "step": step,
                "target_word": word,
                "context_id": context_id,
                "context": f"context-{word}-{context_id}",
                "surprisal": surprisal,
            }

        existing = {
            "results": [
                row(1, "a", 0, 1.0),
                row(1, "b", 0, 2.0),
                row(2, "a", 0, 3.0),
                row(2, "b", 0, float("nan")),
            ]
        }
        kept, completed, rejected = split_complete_resume_results(
            plan, existing, expected_per_step=2
        )

        self.assertEqual(len(kept), 2)
        self.assertEqual(completed, {1})
        self.assertEqual(rejected[2]["finite"], 1)
        self.assertEqual(
            [item["step"] for item in unfinished_steps(plan, completed)], [2, 3]
        )

    def test_validation_rejects_partial_checkpoint(self):
        plan = [{"step": 1}]
        results = {
            "results": [
                {
                    "step": 1,
                    "target_word": "a",
                    "context_id": 0,
                    "context": "context-a",
                    "surprisal": 1.0,
                }
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "incomplete AoA results"):
            validate_complete_results(plan, results, expected_per_step=2)


if __name__ == "__main__":
    unittest.main()
