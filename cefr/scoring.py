from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .alignment import PhraseAlignment
from .data import CEFR_LEVELS, RussianCefrRepository


@dataclass(slots=True)
class CefrScorer:
    repository: RussianCefrRepository
    cefr_order: Sequence[str] = CEFR_LEVELS
    _weights: Mapping[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.cefr_order = tuple(self.cefr_order)
        self._weights = {
            level: idx for idx, level in enumerate(self.cefr_order)
        }

    def infer_level(self, russian_token: str) -> str:
        return self.repository.lookup_level(russian_token)

    def score(self, alignments: Sequence[PhraseAlignment]) -> tuple[dict[str, float], str]:
        counts: Counter[str] = Counter()
        for alignment in alignments:
            level = self.infer_level(alignment.russian_token)
            counts[level] += 1

        known_total = sum(counts[level] for level in self.cefr_order)
        if known_total:
            distribution = {
                level: counts[level] / known_total for level in self.cefr_order
            }
            average_score = sum(
                self._weights[level] * distribution[level]
                for level in self.cefr_order
            )
            average_level = self.cefr_order[round(average_score)]
        else:
            distribution = {level: 0.0 for level in self.cefr_order}
            average_level = "Unknown"
        return distribution, average_level


__all__ = ["CefrScorer"]
