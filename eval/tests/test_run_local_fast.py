import tempfile
import unittest
from pathlib import Path

from eval.run_local_fast import (
    STRICT_REVISIONS,
    STRICT_SMALL_REVISIONS,
    checkpoint_plan,
    required_prediction_files,
)


def manifest_for(revisions):
    checkpoints = []
    for index, revision in enumerate(
        sorted(revisions, key=lambda value: int(value.removeprefix("chck_").removesuffix("M"))),
        start=1,
    ):
        checkpoints.append(
            {
                "path": f"ckpt_{index:06d}.pt",
                "role": "milestone",
                "iter_num": index,
                "tokens_seen": index * 1_000_000,
                "labels": [{"series": "tokens", "revision": revision}],
            }
        )
    return {"checkpoints": checkpoints}


class CheckpointPlanTest(unittest.TestCase):
    def test_strict_small_selects_19_revisions(self):
        plan = checkpoint_plan(Path("/run"), manifest_for(STRICT_REVISIONS), "strict-small")
        self.assertEqual(len(plan), 19)
        self.assertEqual({item["revision"] for item in plan}, STRICT_SMALL_REVISIONS)

    def test_strict_selects_28_revisions(self):
        plan = checkpoint_plan(Path("/run"), manifest_for(STRICT_REVISIONS), "strict")
        self.assertEqual(len(plan), 28)
        self.assertEqual({item["revision"] for item in plan}, STRICT_REVISIONS)

    def test_missing_revision_is_rejected(self):
        revisions = STRICT_SMALL_REVISIONS - {"chck_9M"}
        with self.assertRaisesRegex(ValueError, "chck_9M"):
            checkpoint_plan(Path("/run"), manifest_for(revisions), "strict-small")

    def test_completion_requires_five_core_and_two_global_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            core, global_piqa = required_prediction_files(
                Path(tmp), "model", "chck_1M"
            )
            self.assertEqual(len(core), 5)
            self.assertEqual(len(global_piqa), 2)
            self.assertEqual(len({*core, *global_piqa}), 7)


if __name__ == "__main__":
    unittest.main()
