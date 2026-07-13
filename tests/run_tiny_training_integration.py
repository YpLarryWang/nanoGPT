#!/usr/bin/env python3
"""Run a two-update nanoGPT job and assert the checkpoint budget boundary.

No pytest dependency is required. Run from the repository root with the same
Python interpreter used for formal training.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    dataset_name = "_tiny_checkpoint_integration"
    data_dir = ROOT / "data" / dataset_name
    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True)
    try:
        train = (np.arange(4096, dtype=np.uint16) % 31) + 1
        val = (np.arange(1024, dtype=np.uint16) % 31) + 1
        train.tofile(data_dir / "train.bin")
        val.tofile(data_dir / "val.bin")
        with (data_dir / "meta.pkl").open("wb") as f:
            pickle.dump({"vocab_size": 32, "eot_id": 0, "tokenizer": "synthetic"}, f)

        with tempfile.TemporaryDirectory(prefix="nanogpt-budget-test-") as tmp:
            tmp_dir = Path(tmp)
            out_dir = tmp_dir / "out"
            schedule_path = tmp_dir / "schedule.json"
            log_path = tmp_dir / "experiments.jsonl"
            schedule = {
                "schema_version": 1,
                "parameters": {
                    "max_iters": 2,
                    "block_size": 8,
                    "batch_size": 2,
                    "global_grad_accum": 1,
                    "world_size": 1,
                    "tokens_per_iter": 16,
                    "sampler_seed": 1337,
                },
                "fingerprints": {
                    "train_bin_sha256": sha256(data_dir / "train.bin"),
                    "val_bin_sha256": sha256(data_dir / "val.bin"),
                },
                "cumulative_words": [0, 10, 20],
                "checkpoints": [
                    {
                        "iter_num": 1,
                        "labels": [
                            {"series": "words", "name": "words_test", "revision": "chck_test_words"},
                            {"series": "tokens", "name": "tokens_test", "revision": "chck_test_tokens"},
                        ],
                    },
                    {
                        "iter_num": 2,
                        "labels": [
                            {"series": "words", "name": "words_final", "revision": "chck_final_words"},
                            {"series": "tokens", "name": "tokens_final", "revision": "chck_final_tokens"},
                        ],
                    },
                ],
            }
            schedule_path.write_text(json.dumps(schedule), encoding="utf-8")

            command = [
                sys.executable,
                str(ROOT / "train.py"),
                f"--out_dir={out_dir}",
                f"--dataset={dataset_name}",
                f"--checkpoint_schedule={schedule_path}",
                f"--experiment_log_path={log_path}",
                "--wandb_run_name=checkpoint-integration-b2ga1",
                "--wandb_log=False",
                "--device=cuda" if torch.cuda.is_available() else "--device=cpu",
                "--dtype=float32",
                "--compile=False",
                "--n_layer=1",
                "--n_head=1",
                "--n_embd=16",
                "--block_size=8",
                "--batch_size=2",
                "--gradient_accumulation_steps=1",
                "--eval_batch_size=1",
                "--eval_iters=1",
                "--eval_interval=1",
                "--log_interval=1",
                "--max_iters=2",
                "--lr_decay_iters=2",
                "--warmup_iters=0",
                "--sampler=shuffle",
                "--always_save_checkpoint=False",
            ]
            subprocess.run(command, cwd=ROOT, check=True)

            manifest = json.loads((out_dir / "checkpoint_manifest.json").read_text())
            final_path = out_dir / manifest["roles"]["final"]
            final = torch.load(final_path, map_location="cpu", weights_only=False)
            assert final["iter_num"] == 2
            assert final["num_updates"] == 2
            assert final["checkpoint_role"] == "final"
            assert final["tokens_seen"] == 32
            assert final["words_seen"] == 20
            assert not list(out_dir.glob("*i000003*"))
            milestone = torch.load(out_dir / "ckpt_000001.pt", map_location="cpu", weights_only=False)
            assert {label["series"] for label in milestone["checkpoint_labels"]} == {"words", "tokens"}
            record = json.loads(log_path.read_text().strip())
            assert record["final_iter"] == record["max_iters"] == 2
            assert record["total_tokens"] == 32
            print(f"[ok] exact two-update checkpoint integration: {final_path.name}")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
