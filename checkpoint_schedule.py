"""Validated checkpoint schedules shared by training and offline tooling.

The schedule JSON deliberately keeps *measurement* separate from *policy*:
``cumulative_words`` records what the model actually consumed, while labels say
which word- or token-based milestone caused a snapshot to be retained.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointSchedule:
    path: Path
    max_iters: int
    tokens_per_iter: int
    cumulative_words: tuple[int, ...]
    labels_by_iter: dict[int, tuple[dict, ...]]
    metadata: dict

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        max_iters: int,
        tokens_per_iter: int,
    ) -> "CheckpointSchedule":
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)

        if payload.get("schema_version") != 1:
            raise ValueError(f"unsupported checkpoint schedule schema: {payload.get('schema_version')!r}")

        params = payload.get("parameters", {})
        if params.get("max_iters") != max_iters:
            raise ValueError(
                f"schedule max_iters={params.get('max_iters')} does not match run max_iters={max_iters}"
            )
        if params.get("tokens_per_iter") != tokens_per_iter:
            raise ValueError(
                "schedule tokens_per_iter="
                f"{params.get('tokens_per_iter')} does not match run tokens_per_iter={tokens_per_iter}"
            )

        cumulative_words = tuple(int(x) for x in payload.get("cumulative_words", []))
        if len(cumulative_words) != max_iters + 1:
            raise ValueError(
                f"schedule needs {max_iters + 1} cumulative word counts, got {len(cumulative_words)}"
            )
        if cumulative_words[0] != 0:
            raise ValueError("cumulative_words[0] must be zero")
        if any(b < a for a, b in zip(cumulative_words, cumulative_words[1:])):
            raise ValueError("cumulative_words must be monotonic")

        labels_by_iter: dict[int, list[dict]] = {}
        names: set[tuple[str, str]] = set()
        for item in payload.get("checkpoints", []):
            it = int(item["iter_num"])
            if not 0 <= it <= max_iters:
                raise ValueError(f"checkpoint iter {it} is outside [0, {max_iters}]")
            for label in item.get("labels", []):
                key = (str(label["series"]), str(label["name"]))
                if key in names:
                    raise ValueError(f"duplicate checkpoint label: {key}")
                names.add(key)
                labels_by_iter.setdefault(it, []).append(dict(label))

        return cls(
            path=path,
            max_iters=max_iters,
            tokens_per_iter=tokens_per_iter,
            cumulative_words=cumulative_words,
            labels_by_iter={k: tuple(v) for k, v in labels_by_iter.items()},
            metadata=payload,
        )

    @property
    def save_iters(self) -> set[int]:
        return set(self.labels_by_iter)

    def exposure_at(self, iter_num: int) -> dict[str, int]:
        if not 0 <= iter_num <= self.max_iters:
            raise ValueError(f"iter_num {iter_num} is outside [0, {self.max_iters}]")
        return {
            "tokens_seen": iter_num * self.tokens_per_iter,
            "words_seen": self.cumulative_words[iter_num],
        }

    def labels_at(self, iter_num: int) -> tuple[dict, ...]:
        return self.labels_by_iter.get(iter_num, ())


def rounded_word_tag(words_seen: int | None) -> str:
    """Compact filename tag; exact counts remain in the checkpoint + manifest."""
    if words_seen is None:
        return "wunknown"
    return f"w{round(words_seen / 1_000_000):04d}M"


def at_update_budget(iter_num: int, max_iters: int) -> bool:
    """Return whether training must stop *before* another optimizer update."""
    if iter_num > max_iters:
        raise RuntimeError(f"iter_num={iter_num} exceeded max_iters={max_iters}")
    return iter_num == max_iters
