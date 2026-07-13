import json

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from checkpoint_schedule import CheckpointSchedule, at_update_budget, rounded_word_tag
from data.babylm.build_checkpoint_schedule import (
    EOT,
    SOURCES,
    build_word_starts,
    nearest_iter,
    standard_targets,
)


def test_nearest_iter_prefers_closest_completed_update():
    cumulative = [0, 90, 210, 330]
    assert nearest_iter(cumulative, 100) == 1
    assert nearest_iter(cumulative, 180) == 2


def test_max_iters_is_an_exact_update_count_not_a_zero_based_index():
    iter_num = 0
    updates = 0
    while not at_update_budget(iter_num, 2):
        updates += 1
        iter_num += 1
    assert (updates, iter_num) == (2, 2)
    with pytest.raises(RuntimeError, match="exceeded"):
        at_update_budget(3, 2)


def test_standard_targets_stop_at_actual_exposure():
    targets = standard_targets(89_400_000)
    assert targets[:3] == [1_000_000, 2_000_000, 3_000_000]
    assert targets[-1] == 80_000_000
    assert 90_000_000 not in targets


def test_schedule_validates_and_merges_two_series(tmp_path):
    path = tmp_path / "schedule.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "parameters": {"max_iters": 2, "tokens_per_iter": 100},
                "cumulative_words": [0, 70, 145],
                "checkpoints": [
                    {
                        "iter_num": 1,
                        "labels": [
                            {"series": "words", "name": "words_1M"},
                            {"series": "tokens", "name": "tokens_1M"},
                        ],
                    }
                ],
            }
        )
    )
    schedule = CheckpointSchedule.load(path, max_iters=2, tokens_per_iter=100)
    assert schedule.save_iters == {1}
    assert len(schedule.labels_at(1)) == 2
    assert schedule.exposure_at(2) == {"tokens_seen": 200, "words_seen": 145}


def test_schedule_rejects_update_beyond_budget(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "parameters": {"max_iters": 2, "tokens_per_iter": 100},
                "cumulative_words": [0, 70, 145],
                "checkpoints": [{"iter_num": 3, "labels": []}],
            }
        )
    )
    with pytest.raises(ValueError, match="outside"):
        CheckpointSchedule.load(path, max_iters=2, tokens_per_iter=100)


def test_word_tag_is_human_readable_but_not_the_source_of_truth():
    assert rounded_word_tag(899_466_950) == "w0899M"


def test_word_start_map_matches_clean_whitespace_words(tmp_path):
    data_dir = tmp_path / "data"
    clean_dir = data_dir / "clean" / "train"
    tokenizer_dir = data_dir / "tokenizer"
    clean_dir.mkdir(parents=True)
    tokenizer_dir.mkdir()
    paths = []
    for i, source in enumerate(SOURCES):
        path = clean_dir / f"{source}.txt"
        path.write_text(f"alpha beta {i}\ncan't stop!\n", encoding="utf-8")
        paths.append(str(path))

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train(
        paths,
        BpeTrainer(
            vocab_size=128,
            special_tokens=[EOT],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer_path = tokenizer_dir / "bpe.json"
    tokenizer.save(str(tokenizer_path))
    eot_id = tokenizer.token_to_id(EOT)

    ids = []
    previous = None
    for path in paths:
        for line in open(path, encoding="utf-8"):
            for token_id in tokenizer.encode(line.rstrip("\n")).ids:
                if token_id == eot_id and previous == eot_id:
                    continue
                ids.append(token_id)
                previous = token_id
        if previous != eot_id:
            ids.append(eot_id)
            previous = eot_id
    np.asarray(ids, dtype=np.uint16).tofile(data_dir / "train.bin")

    marks = build_word_starts(data_dir, tokenizer_path, data_dir / "words.uint8")
    assert int(marks.sum()) == len(SOURCES) * 5
