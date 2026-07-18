import tempfile
import unittest
from pathlib import Path

from eval.run_local_aoa import local_step_plan


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


if __name__ == "__main__":
    unittest.main()
