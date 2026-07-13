from pathlib import Path

import pytest

from eval.push_checkpoint_series import role_checkpoint, selected_milestones


def test_selected_milestones_filters_series_and_best():
    manifest = {
        "checkpoints": [
            {
                "role": "milestone",
                "iter_num": 5,
                "labels": [
                    {"series": "words", "revision": "chck_1M"},
                    {"series": "tokens", "revision": "chck_2M"},
                ],
            },
            {
                "role": "best",
                "iter_num": 6,
                "labels": [{"series": "words", "revision": "chck_best"}],
            },
        ]
    }
    selected = selected_milestones(manifest, "words")
    assert len(selected) == 1
    assert selected[0][1]["revision"] == "chck_1M"


def test_role_checkpoint_resolves_existing_file(tmp_path: Path):
    checkpoint = tmp_path / "ckpt_final.pt"
    checkpoint.touch()
    manifest = {"roles": {"final": checkpoint.name}}
    assert role_checkpoint(tmp_path, manifest, "final") == checkpoint


def test_role_checkpoint_rejects_missing_role(tmp_path: Path):
    with pytest.raises(ValueError, match="manifest has no 'best' checkpoint"):
        role_checkpoint(tmp_path, {"roles": {}}, "best")
