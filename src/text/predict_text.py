from __future__ import annotations

from pathlib import Path

from src.align.mutual_align import EmbeddingAligner
from src.data.repositories import DEFAULT_RUS_CEFR, RussianCefrRepository
from src.domain.entities import TextCefrPrediction
from src.domain.services import (
    AlignmentService,
    CefrScorer,
    TextCefrPipeline,
    TranslationService,
)
from src.translation.translator import Translator


def predict_text_cefr(
    kaz_text: str,
    rus_cefr_path: str | Path = DEFAULT_RUS_CEFR,
    translator: Translator | None = None,
    aligner: EmbeddingAligner | None = None,
    *,
    russian_text: str | None = None,
) -> TextCefrPrediction:
    translation_service = TranslationService(translator)
    alignment_service = AlignmentService(aligner)
    scorer = CefrScorer(RussianCefrRepository(rus_cefr_path))
    pipeline = TextCefrPipeline(
        translation_service=translation_service,
        alignment_service=alignment_service,
        scorer=scorer,
    )
    return pipeline.predict(kaz_text, russian_text=russian_text)
