from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


CEFR_ORDER: tuple[str, ...] = ("A1", "A2", "B1", "B2", "C1", "C2")


@dataclass(frozen=True)
class PhraseAlignment:
    """Represents an aligned Kazakh span to a single Russian token."""

    kazakh_phrase: str
    russian_token: str
    kazakh_span: tuple[int, ...]
    russian_index: int


@dataclass(frozen=True)
class WordAlignment:
    """Atomic Kazakh↔Russian alignment with CEFR metadata."""

    kazakh_token: str
    russian_token: str
    kazakh_index: int
    russian_index: int
    cefr: str

    def as_tuple(self) -> tuple[str, str, int, int, str]:
        return (
            self.kazakh_token,
            self.russian_token,
            self.kazakh_index,
            self.russian_index,
            self.cefr,
        )


@dataclass(frozen=True)
class TextCefrPrediction:
    """Aggregated CEFR metadata for a Kazakh text sample."""

    translation: str
    distribution: Mapping[str, float]
    average_level: str
    phrase_alignments: Sequence[PhraseAlignment] = field(default_factory=tuple)
    word_alignments: Sequence[WordAlignment] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Serialize the prediction to a plain dictionary."""

        payload = {
            "translation": self.translation,
            "distribution": dict(self.distribution),
            "avg_level": self.average_level,
            "phrases": [
                (
                    phrase.kazakh_phrase,
                    phrase.russian_token,
                    phrase.kazakh_span,
                    phrase.russian_index,
                )
                for phrase in self.phrase_alignments
            ],
            "word_alignments": [
                alignment.as_tuple() for alignment in self.word_alignments
            ],
        }
        return payload

    def iter_word_alignment_tuples(self) -> Iterable[tuple[str, str, int, int, str]]:
        for alignment in self.word_alignments:
            yield alignment.as_tuple()

    @property
    def word_alignment_tuples(self) -> list[tuple[str, str, int, int, str]]:
        return [alignment.as_tuple() for alignment in self.word_alignments]


__all__ = ["PhraseAlignment", "WordAlignment", "TextCefrPrediction", "CEFR_ORDER"]
