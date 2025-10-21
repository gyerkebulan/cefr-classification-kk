from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence
import warnings

from src.align.merge_phrases import merge_kz_to_single_ru
from src.align.mutual_align import (
    EmbeddingAligner,
    SequenceTooLongError,
    get_default_aligner,
)
from src.domain.entities import CEFR_ORDER, PhraseAlignment, TextCefrPrediction
from src.data.repositories import RussianCefrRepository
from src.translation.translator import Translator, get_translator


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(part for part in text.strip().split() if part)


class TranslationService:
    """Facade over the HuggingFace translation pipeline."""

    def __init__(self, translator: Translator | None = None) -> None:
        self._translator: Translator | None = translator

    def translate(self, text: str, *, override: str | None = None) -> str:
        if override is not None:
            return override
        if self._translator is None:
            self._translator = get_translator()
        return self._translator.translate(text)


class AlignmentService:
    """Align Kazakh → Russian tokens and merge spans into phrases."""

    def __init__(
        self,
        aligner: EmbeddingAligner | None = None,
        *,
        default_layer: int = 8,
        default_threshold: float = 0.05,
    ) -> None:
        self._aligner = aligner or get_default_aligner()
        self._layer = default_layer
        self._threshold = default_threshold

    def align_phrases(
        self,
        kazakh_words: Sequence[str],
        russian_words: Sequence[str],
        *,
        layer: int | None = None,
        threshold: float | None = None,
    ) -> list[PhraseAlignment]:
        try:
            links = self._aligner.align(
                kazakh_words,
                russian_words,
                layer=layer if layer is not None else self._layer,
                thresh=threshold if threshold is not None else self._threshold,
            )
        except SequenceTooLongError as exc:
            warnings.warn(
                f"Skipping alignment for sequence exceeding max length: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return []
        return merge_kz_to_single_ru(kazakh_words, russian_words, links)


class CefrScorer:
    """Aggregate CEFR statistics for aligned Russian tokens."""

    def __init__(
        self,
        repository: RussianCefrRepository,
        *,
        cefr_order: Sequence[str] = CEFR_ORDER,
    ) -> None:
        self._repository = repository
        self._cefr_order = tuple(cefr_order)
        self._weights: Mapping[str, int] = {level: idx for idx, level in enumerate(self._cefr_order)}

    def infer_level(self, russian_token: str) -> str:
        return self._repository.lookup_level(russian_token)

    def score_alignments(self, alignments: Sequence[PhraseAlignment]) -> tuple[dict[str, float], str]:
        counts: Counter[str] = Counter()
        for alignment in alignments:
            level = self.infer_level(alignment.russian_token)
            counts[level] += 1

        known_total = sum(counts[level] for level in self._cefr_order)
        if known_total:
            distribution = {
                level: counts[level] / known_total for level in self._cefr_order
            }
            average_score = sum(
                self._weights[level] * distribution[level] for level in self._cefr_order
            )
            average_level = self._cefr_order[round(average_score)]
        else:
            distribution = {level: 0.0 for level in self._cefr_order}
            average_level = "Unknown"

        return distribution, average_level


class TextCefrPipeline:
    """High-level use-case orchestrating translation, alignment and CEFR scoring."""

    def __init__(
        self,
        translation_service: TranslationService | None = None,
        alignment_service: AlignmentService | None = None,
        scorer: CefrScorer | None = None,
    ) -> None:
        self._translation = translation_service or TranslationService()
        self._alignment = alignment_service or AlignmentService()
        if scorer is None:
            scorer = CefrScorer(RussianCefrRepository())
        self._scorer = scorer

    def predict(self, kazakh_text: str, *, russian_text: str | None = None) -> TextCefrPrediction:
        translation = self._translation.translate(kazakh_text, override=russian_text)
        kazakh_words = _tokenize(kazakh_text)
        russian_words = _tokenize(translation)
        alignments = self._alignment.align_phrases(kazakh_words, russian_words)
        distribution, avg_level = self._scorer.score_alignments(alignments)
        return TextCefrPrediction(
            translation=translation,
            distribution=distribution,
            average_level=avg_level,
            phrase_alignments=alignments,
        )


__all__ = [
    "TranslationService",
    "AlignmentService",
    "CefrScorer",
    "TextCefrPipeline",
]
