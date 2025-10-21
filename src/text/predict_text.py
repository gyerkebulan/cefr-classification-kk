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
from src.text.ensemble_pipeline import EnsembleCefrPipeline
from src.ru_sentence_model.model import RuSentenceCefrModel


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


def predict_text_cefr_ensemble(
    kaz_text: str,
    *,
    rus_cefr_path: str | Path = DEFAULT_RUS_CEFR,
    translator: Translator | None = None,
    aligner: EmbeddingAligner | None = None,
    russian_text: str | None = None,
    russian_model_checkpoint: str | Path | None = None,
    russian_model: RuSentenceCefrModel | None = None,
    russian_weight: float = 0.6,
) -> dict[str, object]:
    translation_service = TranslationService(translator)
    alignment_service = AlignmentService(aligner)
    scorer = CefrScorer(RussianCefrRepository(rus_cefr_path))
    base_pipeline = TextCefrPipeline(
        translation_service=translation_service,
        alignment_service=alignment_service,
        scorer=scorer,
    )
    if russian_model is None:
        if russian_model_checkpoint is None:
            raise ValueError("Either russian_model or russian_model_checkpoint must be provided.")
        russian_model = RuSentenceCefrModel.from_pretrained(russian_model_checkpoint)
    ensemble = EnsembleCefrPipeline(
        kazakh_pipeline=base_pipeline,
        russian_model=russian_model,
        russian_weight=russian_weight,
    )
    prediction = ensemble.predict(kaz_text, russian_text=russian_text)
    return {
        "level": prediction.level,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "translation": prediction.translation,
        "kazakh_distribution": prediction.kazakh_distribution,
        "russian_distribution": prediction.russian_distribution,
        "word_alignments": prediction.word_alignments,
    }
