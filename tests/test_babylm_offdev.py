from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BABYLM = ROOT / "data" / "babylm"
sys.path.insert(0, str(BABYLM))

import constants  # noqa: E402
import fetch_offdev  # noqa: E402
import train_bpe  # noqa: E402


class OffdevProtocolTest(unittest.TestCase):
    def test_release_word_budgets_are_locked(self):
        self.assertEqual(sum(fetch_offdev.RELEASES["10m"]["words"].values()), 10_000_000)
        self.assertEqual(sum(fetch_offdev.RELEASES["100m"]["words"].values()), 100_000_000)
        self.assertEqual(sum(fetch_offdev.DEV_RELEASE["words"].values()), 10_418_979)

    def test_clean_cli_uses_dev_filename_without_txt_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            out = root / "clean"
            raw.mkdir()
            (raw / "bnc_spoken.dev").write_text("A development sentence.\n", encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(BABYLM / "clean.py"),
                    "--input-split=dev",
                    f"--raw-dir={raw}",
                    f"--out-dir={out}",
                    "--sources=bnc_spoken",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                (out / "bnc_spoken.txt").read_text(encoding="utf-8"),
                "A development sentence.\n",
            )

    def test_tokenizer_paths_are_train_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean = Path(tmp)
            (clean / "train").mkdir()
            (clean / "val").mkdir()
            for source in constants.SOURCES:
                (clean / "train" / f"{source}.txt").write_text("train text\n", encoding="utf-8")
                (clean / "val" / f"{source}.txt").write_text("dev text\n", encoding="utf-8")
            paths = train_bpe.train_paths(str(clean), constants.SOURCES)
            self.assertEqual(len(paths), len(constants.SOURCES))
            self.assertTrue(all("/train/" in path for path in paths))
            self.assertTrue(all("/val/" not in path for path in paths))

    def test_tiny_offdev_pipeline_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "babylm_officialdev"
            records = {}
            for raw_split in ("train", "dev"):
                raw_dir = data_dir / "raw" / raw_split
                raw_dir.mkdir(parents=True)
                rows = []
                for source in constants.SOURCES:
                    filename = constants.RAW_FILENAME_TEMPLATES[raw_split].format(source=source)
                    path = raw_dir / filename
                    path.write_text(
                        f"{raw_split} sentence for {source}.\n"
                        f"another {raw_split} example from {source}.\n",
                        encoding="utf-8",
                    )
                    lines, words = fetch_offdev.text_counts(path)
                    rows.append({
                        "source": source,
                        "filename": filename,
                        "url": "test://fixture",
                        "bytes": path.stat().st_size,
                        "lines": lines,
                        "words": words,
                        "sha256": fetch_offdev.sha256_file(path),
                    })
                records[raw_split] = rows
            source_manifest = {
                "schema_version": 1,
                "protocol": "official-train-dev-v1",
                "track": "10m",
                "splits": {
                    split: {"repo": "test", "revision": "test", "files": rows}
                    for split, rows in records.items()
                },
            }
            (data_dir / "source_manifest.json").write_text(
                json.dumps(source_manifest), encoding="utf-8"
            )

            commands = [
                [
                    sys.executable, str(BABYLM / "clean.py"), "--input-split=train",
                    f"--raw-dir={data_dir / 'raw' / 'train'}",
                    f"--out-dir={data_dir / 'clean' / 'train'}",
                ],
                [
                    sys.executable, str(BABYLM / "clean.py"), "--input-split=dev",
                    f"--raw-dir={data_dir / 'raw' / 'dev'}",
                    f"--out-dir={data_dir / 'clean' / 'val'}",
                ],
                [
                    sys.executable, str(BABYLM / "train_bpe.py"),
                    f"--clean-dir={data_dir / 'clean'}",
                    f"--out-dir={data_dir / 'tokenizer'}", "--vocab-sizes=300",
                ],
                [
                    sys.executable, str(BABYLM / "prepare.py"),
                    f"--data-dir={data_dir}", "--tokenizer=bpe-300.json",
                ],
                [sys.executable, str(BABYLM / "audit_offdev.py"), f"--data-dir={data_dir}"],
            ]
            for command in commands:
                subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

            audit = json.loads((data_dir / "audit_report.json").read_text(encoding="utf-8"))
            self.assertTrue(audit["tokenizer_provenance"]["train_only"])
            self.assertEqual(
                audit["leakage_audit"]["totals"]["exact_unique_long_line_overlaps"], 0
            )
            self.assertGreater(audit["training_budget"]["train_tokens"], 0)
            data_manifest = json.loads((data_dir / "data_manifest.json").read_text())
            self.assertEqual(data_manifest["protocol"], "official-train-dev-v1")

    def test_formal_pipeline_modules_do_not_import_legacy_split(self):
        for filename in (
            "clean.py",
            "train_bpe.py",
            "prepare.py",
            "build_checkpoint_schedule.py",
            "fetch_offdev.py",
            "audit_offdev.py",
        ):
            source = (BABYLM / filename).read_text(encoding="utf-8")
            self.assertNotIn("import split", source, filename)
            self.assertNotIn("from split import", source, filename)

    def test_offdev_wandb_run_name_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "must contain '-offdev'"):
            constants.validate_offdev_wandb_name(
                "babylm_officialdev", True, "missing-protocol-tag"
            )
        constants.validate_offdev_wandb_name(
            "babylm_officialdev", True, "bl10m-d512L32-offdev"
        )
        constants.validate_offdev_wandb_name(
            "babylm_officialdev", False, "smoke"
        )


if __name__ == "__main__":
    unittest.main()
