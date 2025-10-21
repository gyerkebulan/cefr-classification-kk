from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.domain.entities import TextCefrPrediction
from src.domain.services import TextCefrPipeline
from src.ru_sentence_model.model import RuSentenceCefrModel
from src.ru_sentence_model.data import CEFR_LEVELS


@dataclass(slots=True)
class EnsemblePrediction:
    level: str
    probabilities: Mapping[str, float]
    confidence: float
    translation: str
    kazakh_distribution: Mapping[str, float]
    russian_distribution: Mapping[str, float]
    word_alignments: Sequence[tuple[str, str, int, int, str]]
    base_prediction: TextCefrPrediction


class EnsembleCefrPipeline:
    def __init__(
        self,
        *,
        kazakh_pipeline: TextCefrPipeline,
        russian_model: RuSentenceCefrModel,
        russian_weight: float = 0.6,
    ) -> None:
        if not 0.0 <= russian_weight <= 1.0:
            raise ValueError("russian_weight must be between 0 and 1 inclusive.")
        self._kk_pipeline = kazakh_pipeline
        self._ru_model = russian_model
        self._weight = russian_weight

    def _weighted_average(
        self, ru_probs: Mapping[str, float], kk_probs: Mapping[str, float]
    ) -> dict[str, float]:
        combined = {}
        for level in CEFR_LEVELS:
            combined[level] = (
                self._weight * ru_probs.get(level, 0.0)
                + (1.0 - self._weight) * kk_probs.get(level, 0.0)
            )
        total = sum(combined.values())
        if total <= 0:
            uniform = 1.0 / len(CEFR_LEVELS)
            return {level: uniform for level in CEFR_LEVELS}
        return {level: prob / total for level, prob in combined.items()}

    def predict(
        self,
        kazakh_text: str,
        *,
        russian_text: str | None = None,
    ) -> EnsemblePrediction:
        base_prediction = self._kk_pipeline.predict(kazakh_text, russian_text=russian_text)
        translation = base_prediction.translation
        ru_probs = self._ru_model.predict_proba(translation)
        kk_probs = dict(base_prediction.distribution)
        combined = self._weighted_average(ru_probs, kk_probs)
        level = max(combined, key=combined.get)
        confidence = combined[level]
        word_alignments = list(base_prediction.word_alignment_tuples)
        return EnsemblePrediction(
            level=level,
            probabilities=combined,
            confidence=confidence,
            translation=translation,
            kazakh_distribution=kk_probs,
            russian_distribution=ru_probs,
            word_alignments=word_alignments,
            base_prediction=base_prediction,
        )


__all__ = ["EnsemblePrediction", "EnsembleCefrPipeline"]
