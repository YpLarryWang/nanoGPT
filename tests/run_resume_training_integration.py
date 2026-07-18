#!/usr/bin/env python3
"""Prove an interrupted shuffle run resumes to the exact uninterrupted state.

The test runs against a temporary copy of the training runtime. A test-only
injection exits immediately after the iter-2 full checkpoint, then the normal
resume path continues to iter 4. No repository data or outputs are modified.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


SOURCE = Path(__file__).resolve().parents[1]


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def assert_equal(left, right, path="root"):
    if torch.is_tensor(left):
        assert torch.equal(left, right), path
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            assert_equal(left_item, right_item, f"{path}[{index}]")
    else:
        assert left == right, path


def copy_runtime(destination: Path):
    for name in (
        "train.py",
        "configurator.py",
        "model.py",
        "masked_data.py",
        "checkpoint_schedule.py",
    ):
        shutil.copy2(SOURCE / name, destination / name)
    babylm = destination / "data" / "babylm"
    babylm.mkdir(parents=True)
    shutil.copy2(SOURCE / "data" / "babylm" / "constants.py", babylm / "constants.py")
    (destination / "data" / "__init__.py").touch()
    (babylm / "__init__.py").touch()

    train_path = destination / "train.py"
    source = train_path.read_text(encoding="utf-8")
    import_anchor = "from torch.distributed import init_process_group, destroy_process_group\n"
    seeded = "\nrandom.seed(1339)\nnp.random.seed(1339)\n"
    loop_anchor = "    # iter_num is completed optimizer updates. W(max_iters) is eligible for best,\n"
    interrupted = (
        "    test_stop = os.environ.get('NANOGPT_TEST_STOP_ITER')\n"
        "    if test_stop is not None and iter_num == int(test_stop):\n"
        "        raise SystemExit(99)\n\n"
    )
    assert import_anchor in source and loop_anchor in source
    source = source.replace(import_anchor, import_anchor + seeded, 1)
    source = source.replace(loop_anchor, interrupted + loop_anchor, 1)
    train_path.write_text(source, encoding="utf-8")


def make_data(root: Path):
    data = root / "data" / "resume_integration"
    data.mkdir()
    rng = np.random.default_rng(20260718)
    rng.integers(0, 128, size=8192, dtype=np.uint16).tofile(data / "train.bin")
    rng.integers(0, 128, size=2048, dtype=np.uint16).tofile(data / "val.bin")
    with (data / "meta.pkl").open("wb") as handle:
        pickle.dump({"vocab_size": 128}, handle)


def run(root: Path, command, *, stop=None, expected=0):
    environment = os.environ.copy()
    if stop is not None:
        environment["NANOGPT_TEST_STOP_ITER"] = str(stop)
    result = subprocess.run(command, cwd=root, env=environment)
    assert result.returncode == expected, (result.returncode, expected)


def final_checkpoint(out_dir: Path):
    manifest = json.loads((out_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    return load(out_dir / manifest["roles"]["final"]), manifest


def main():
    with tempfile.TemporaryDirectory(prefix="nanogpt-resume-integration-") as temporary:
        root = Path(temporary)
        copy_runtime(root)
        make_data(root)
        continuous = root / "continuous"
        resumed = root / "resumed"
        common = [
            sys.executable,
            "train.py",
            "--dataset=resume_integration",
            "--device=cpu",
            "--dtype=float32",
            "--compile=False",
            "--n_layer=2",
            "--n_head=4",
            "--n_embd=32",
            "--block_size=8",
            "--dropout=0.1",
            "--batch_size=2",
            "--gradient_accumulation_steps=2",
            "--sampler=shuffle",
            "--sampler_seed=1339",
            "--seed=1339",
            "--max_iters=4",
            "--lr_decay_iters=4",
            "--warmup_iters=1",
            "--eval_interval=1",
            "--eval_iters=2",
            "--eval_batch_size=2",
            "--always_save_checkpoint=True",
            "--wandb_log=False",
            "--log_interval=1",
        ]
        run(
            root,
            common
            + [f"--out_dir={continuous}", f"--experiment_log_path={root / 'continuous.jsonl'}"],
        )
        run(
            root,
            common + [f"--out_dir={resumed}", f"--experiment_log_path={root / 'resumed.jsonl'}"],
            stop=2,
            expected=99,
        )
        resume_checkpoint = resumed / "ckpt_latest.pt"
        assert load(resume_checkpoint)["iter_num"] == 2
        run(
            root,
            common
            + [
                f"--out_dir={resumed}",
                f"--experiment_log_path={root / 'resumed.jsonl'}",
                "--init_from=resume",
                f"--resume_checkpoint={resume_checkpoint}",
            ],
        )

        uninterrupted, uninterrupted_manifest = final_checkpoint(continuous)
        recovered, recovered_manifest = final_checkpoint(resumed)
        for key in ("model", "optimizer", "prefetched_batch", "rng_state"):
            assert_equal(uninterrupted[key], recovered[key], key)
        assert uninterrupted["final_train_loss"] == recovered["final_train_loss"]
        assert uninterrupted["final_val_loss"] == recovered["final_val_loss"]
        assert len(uninterrupted_manifest["checkpoints"]) == len(recovered_manifest["checkpoints"])
        print("[ok] interrupted resume is bitwise-identical to uninterrupted training")


if __name__ == "__main__":
    main()
