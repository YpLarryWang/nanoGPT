from eval.push_checkpoint_series import selected_milestones


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
