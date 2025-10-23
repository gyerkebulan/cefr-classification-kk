from collections import Counter

from .data import CEFR_LEVELS, RussianCefrRepository
from .text_utils import is_cyrillic_token


class CefrScorer:
    __slots__ = ("repository", "cefr_order", "_weights")

    def __init__(self, repository, cefr_order=CEFR_LEVELS):
        self.repository = repository
        self.cefr_order = tuple(cefr_order)
        self._weights = {level: idx for idx, level in enumerate(self.cefr_order)}

    def infer_level(self, russian_token):
        return self.repository.lookup_level(russian_token)

    def score(self, alignments):
        counts = Counter()
        for alignment in alignments:
            token = alignment.russian_token
            if not is_cyrillic_token(token):
                continue
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
