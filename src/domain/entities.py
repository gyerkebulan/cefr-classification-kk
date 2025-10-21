from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


CEFR_ORDER: tuple[str, ...] = ("A1", "A2", "B1", "B2", "C1", "C2")


@dataclass(frozen=True)
class PhraseAlignment:
    """Represents an aligned Kazakh span to a single Russian token."""

    kazakh_phrase: str
    russian_token: str
    kazakh_span: tuple[int, ...]
    russian_index: int


@dataclass(frozen=True)
class TextCefrPrediction:
    """Aggregated CEFR metadata for a Kazakh text sample."""

    translation: str
    distribution: Mapping[str, float]
    average_level: str
    phrase_alignments: Sequence[PhraseAlignment] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Serialize the prediction to a plain dictionary."""

        return {
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
        }


__all__ = ["PhraseAlignment", "TextCefrPrediction", "CEFR_ORDER"]
