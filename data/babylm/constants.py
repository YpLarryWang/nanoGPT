"""Shared constants for the BabyLM data pipeline.

This module is intentionally split-agnostic.  New official-dev datasets must
not import the legacy tail-carving implementation in ``split.py``.
"""

SOURCES = [
    "bnc_spoken",
    "childes",
    "gutenberg",
    "open_subtitles",
    "simple_wiki",
    "switchboard",
]

EOT = "<|endoftext|>"

RAW_FILENAME_TEMPLATES = {
    "train": "{source}.train.txt",
    "dev": "{source}.dev",
    "test": "{source}.test",
}

CLEAN_SPLIT_NAMES = {
    "train": "train",
    "dev": "val",
    "test": "test",
}

OFFDEV_DATASETS = {"babylm_officialdev", "babylm_100m_officialdev"}


def validate_offdev_wandb_name(dataset: str, wandb_log: bool, run_name: str) -> None:
    """Fail closed when a formal offdev W&B run lacks its protocol tag."""
    if dataset in OFFDEV_DATASETS and wandb_log and "-offdev" not in run_name:
        raise ValueError(f"official-dev W&B run name must contain '-offdev': {run_name!r}")


def validate_offdev_checkpoint_schedule(
    dataset: str, wandb_log: bool, checkpoint_schedule: str
) -> None:
    """Require AoA checkpoint coverage for every formal offdev run."""
    if dataset in OFFDEV_DATASETS and wandb_log and not checkpoint_schedule:
        raise ValueError("official-dev W&B runs must set checkpoint_schedule for AoA")
