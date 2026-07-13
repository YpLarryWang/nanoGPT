from pathlib import Path

from eval.run_local_aoa import local_step_plan


def test_local_step_plan_uses_target_counts_and_word_series(tmp_path: Path):
    manifest = {
        "checkpoints": [
            {
                "role": "milestone",
                "path": "ckpt_000005.pt",
                "iter_num": 5,
                "sha256": "abc",
                "labels": [
                    {
                        "series": "words",
                        "revision": "chck_1M",
                        "target": 1_000_000,
                        "actual": 958_569,
                    },
                    {
                        "series": "tokens",
                        "revision": "chck_1M",
                        "target": 1_000_000,
                        "actual": 1_310_720,
                    },
                ],
            }
        ]
    }

    assert local_step_plan(tmp_path, manifest, "words") == [
        {
            "step": 1_000_000,
            "revision": "chck_1M",
            "source": tmp_path / "ckpt_000005.pt",
            "iter_num": 5,
            "actual": 958_569,
            "sha256": "abc",
        }
    ]
