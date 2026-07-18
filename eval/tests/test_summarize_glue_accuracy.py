import tempfile
import unittest
from pathlib import Path

from eval.summarize_glue_accuracy import TASKS, collect, read_accuracy


class SummarizeGlueAccuracyTest(unittest.TestCase):
    def test_collect_uses_accuracy_for_every_task(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "results" / "model" / "main" / "finetune"
            for index, task in enumerate(TASKS):
                path = base / task / "results.txt"
                path.parent.mkdir(parents=True)
                path.write_text(f"accuracy: {0.5 + index * 0.01}\nf1: 0.99\n")

            summary = collect(root, "model")
            self.assertEqual(summary["metric_protocol"], "all_accuracy")
            self.assertEqual(summary["task_accuracy"]["mrpc"], 54.0)
            self.assertEqual(summary["task_accuracy"]["qqp"], 55.0)
            self.assertEqual(summary["superglue_ld"], 53.0)

    def test_rejects_out_of_range_accuracy(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.txt"
            path.write_text("accuracy: 1.2\n")
            with self.assertRaises(ValueError):
                read_accuracy(path)


if __name__ == "__main__":
    unittest.main()
