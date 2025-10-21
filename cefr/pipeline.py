from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from cefr.alignment import (
    EmbeddingAligner,
    PhraseAlignment,
    WordAlignment,
    merge_kz_to_single_ru,
)
from cefr.config import AlignmentConfig, PipelineConfig, TranslatorConfig
from cefr.data import RussianCefrRepository
from cefr.models import RuSentenceCefrModel
from cefr.scoring import CefrScorer
from cefr.translation import Translator, get_translator


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(part for part in text.strip().split() if part)


@dataclass(slots=True)
class TextPrediction:
    translation: str
    distribution: Mapping[str, float]
    average_level: str
    phrase_alignments: Sequence[PhraseAlignment]
    word_alignments: Sequence[WordAlignment]

    def iter_word_alignment_tuples(self) -> Iterable[tuple[str, str, int, int, str]]:
        for alignment in self.word_alignments:
            yield alignment.as_tuple()

    @property
    def word_alignment_tuples(self) -> list[tuple[str, str, int, int, str]]:
        return list(self.iter_word_alignment_tuples())


class TextPipeline:
    """Translate, align and score Kazakh text."""

    def __init__(
        self,
        *,
        config: PipelineConfig | None = None,
        translator: Translator | None = None,
        aligner: EmbeddingAligner | None = None,
        scorer: CefrScorer | None = None,
        repository: RussianCefrRepository | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.translator = translator or get_translator(self.config.translator)
        aligner_config = self.config.alignment
        self.aligner = aligner or EmbeddingAligner(aligner_config)
        repo_path = repository.path if repository else self.config.russian_cefr_path
        self.repository = repository or RussianCefrRepository(Path(repo_path))
        self.scorer = scorer or CefrScorer(self.repository)

    def predict(self, kazakh_text: str, *, russian_text: str | None = None) -> TextPrediction:
        translation = russian_text or self.translator.translate(kazakh_text)
        kazakh_words = _tokenize(kazakh_text)
        russian_words = _tokenize(translation)
        links = self.aligner.align(kazakh_words, russian_words)
        phrases = merge_kz_to_single_ru(kazakh_words, russian_words, links)
        distribution, avg_level = self.scorer.score(phrases)
        word_alignments = self._build_word_alignments(phrases, kazakh_words)
        return TextPrediction(
            translation=translation,
            distribution=distribution,
            average_level=avg_level,
            phrase_alignments=tuple(phrases),
            word_alignments=tuple(word_alignments),
        )

    def _build_word_alignments(
        self, phrases: Sequence[PhraseAlignment], kazakh_words: Sequence[str]
    ) -> list[WordAlignment]:
        word_alignments: list[WordAlignment] = []
        for phrase in phrases:
            level = self.scorer.infer_level(phrase.russian_token)
            for kaz_idx in phrase.kazakh_span:
                if 0 <= kaz_idx < len(kazakh_words):
                    word_alignments.append(
                        WordAlignment(
                            kazakh_token=kazakh_words[kaz_idx],
                            russian_token=phrase.russian_token,
                            kazakh_index=kaz_idx,
                            russian_index=phrase.russian_index,
                            cefr=level,
                        )
                    )
        return word_alignments


@dataclass(slots=True)
class EnsemblePrediction:
    level: str
    confidence: float
    probabilities: Mapping[str, float]
    translation: str
    kazakh_distribution: Mapping[str, float]
    russian_distribution: Mapping[str, float]
    word_alignments: Sequence[tuple[str, str, int, int, str]]
    base_prediction: TextPrediction


class EnsemblePipeline:
    def __init__(
        self,
        *,
        base_pipeline: TextPipeline,
        russian_model: RuSentenceCefrModel,
        russian_weight: float = 0.6,
    ) -> None:
        if not 0.0 <= russian_weight <= 1.0:
            raise ValueError("russian_weight must be between 0 and 1 inclusive.")
        self.base = base_pipeline
        self.russian_model = russian_model
        self.weight = russian_weight

    def _weighted_average(
        self, ru_probs: Mapping[str, float], kk_probs: Mapping[str, float]
    ) -> dict[str, float]:
        combined = {}
        for level in self.base.scorer.cefr_order:
            combined[level] = (
                self.weight * ru_probs.get(level, 0.0)
                + (1.0 - self.weight) * kk_probs.get(level, 0.0)
            )
        total = sum(combined.values())
        if total <= 0:
            uniform = 1.0 / len(combined)
            return {level: uniform for level in combined}
        return {level: value / total for level, value in combined.items()}

    def predict(self, kazakh_text: str, *, russian_text: str | None = None) -> EnsemblePrediction:
        base_prediction = self.base.predict(kazakh_text, russian_text=russian_text)
        translation = base_prediction.translation
        ru_probs = self.russian_model.predict_proba(translation)
        kk_probs = dict(base_prediction.distribution)
        combined = self._weighted_average(ru_probs, kk_probs)
        level = max(combined, key=combined.get)
        confidence = combined[level]
        return EnsemblePrediction(
            level=level,
            confidence=confidence,
            probabilities=combined,
            translation=translation,
            kazakh_distribution=kk_probs,
            russian_distribution=ru_probs,
            word_alignments=base_prediction.word_alignment_tuples,
            base_prediction=base_prediction,
        )


__all__ = [
    "TextPipeline",
    "TextPrediction",
    "EnsemblePipeline",
    "EnsemblePrediction",
]
