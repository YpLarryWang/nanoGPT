from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from full_dev_loss import (  # noqa: E402
    OFFICIAL_TEST_REPO,
    OFFICIAL_TEST_REVISION,
    evaluate_model,
    sha256_file,
    token_batches,
    validate_test_data,
)


class MeanTargetModel(torch.nn.Module):
    def forward(self, x, targets=None):
        del x
        return torch.empty(0), targets.float().mean()


class FullDevLossTest(unittest.TestCase):
    def test_token_batches_cover_every_next_token_once(self):
        data = np.arange(19, dtype=np.uint16)
        observed_x = []
        observed_y = []
        shapes = []
        for x, y in token_batches(data, block_size=4, batch_size=2):
            shapes.append((tuple(x.shape), tuple(y.shape)))
            observed_x.extend(x.reshape(-1).tolist())
            observed_y.extend(y.reshape(-1).tolist())
        self.assertEqual(observed_x, list(range(18)))
        self.assertEqual(observed_y, list(range(1, 19)))
        self.assertEqual(shapes, [((2, 4), (2, 4)), ((2, 4), (2, 4)), ((1, 2), (1, 2))])

    def test_evaluate_model_weights_by_target_token_not_batch(self):
        data = np.arange(12, dtype=np.uint16)
        loss, tokens, _ = evaluate_model(
            model=MeanTargetModel(),
            data=data,
            block_size=4,
            batch_size=2,
            device=torch.device("cpu"),
            autocast_dtype=None,
            log_every=0,
        )
        self.assertEqual(tokens, 11)
        self.assertAlmostEqual(loss, float(np.arange(1, 12).mean()), places=7)

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.bin"
            path.write_bytes(b"BabyLM full dev")
            self.assertEqual(sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_rejects_too_short_data(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            list(token_batches(np.array([1], dtype=np.uint16), 4, 2))

    def test_test_manifest_must_match_checkpoint_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            data_path = data_dir / "test.bin"
            data_path.write_bytes(b"\x01\x00\x02\x00")
            source_manifest_path = data_dir / "test_source_manifest.json"
            source_manifest_path.write_text(
                json.dumps(
                    {
                        "protocol": "official-test-v1",
                        "repo": OFFICIAL_TEST_REPO,
                        "revision": OFFICIAL_TEST_REVISION,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            test_manifest = {
                "protocol": "official-test-tokenized-v1",
                "repo": OFFICIAL_TEST_REPO,
                "revision": OFFICIAL_TEST_REVISION,
                "source_manifest_sha256": sha256_file(source_manifest_path),
                "tokenizer": {"sha256": "test-tokenizer"},
                "bin": {"sha256": sha256_file(data_path), "tokens": 2},
            }
            (data_dir / "test_manifest.json").write_text(
                json.dumps(test_manifest, sort_keys=True), encoding="utf-8"
            )
            checkpoint = {
                "provenance": {
                    "data_fingerprints": {"tokenizer_sha256": "test-tokenizer"}
                }
            }
            validated = validate_test_data(checkpoint, data_dir, data_path)
            self.assertEqual(validated["official_revision"], OFFICIAL_TEST_REVISION)

            checkpoint["provenance"]["data_fingerprints"]["tokenizer_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "tokenizer SHA-256 mismatch"):
                validate_test_data(checkpoint, data_dir, data_path)


if __name__ == "__main__":
    unittest.main()
