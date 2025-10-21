from __future__ import annotations

from pathlib import Path

from cefr.config import PipelineConfig
from cefr.data import RussianCefrRepository
from cefr.models import RuSentenceCefrModel
from cefr.pipeline import EnsemblePipeline, TextPipeline
from cefr.translation import Translator
from cefr.alignment import EmbeddingAligner


def _build_pipeline(
    rus_cefr_path: Path,
    translator: Translator | None,
    aligner: EmbeddingAligner | None,
) -> TextPipeline:
    config = PipelineConfig(russian_cefr_path=str(rus_cefr_path))
    repository = RussianCefrRepository(rus_cefr_path)
    return TextPipeline(
        config=config,
        translator=translator,
        aligner=aligner,
        repository=repository,
    )


def predict_text_cefr(
    kaz_text: str,
    rus_cefr_path: str | Path = Path("data/cefr/russian_cefr_sample.csv"),
    translator: Translator | None = None,
    aligner: EmbeddingAligner | None = None,
    *,
    russian_text: str | None = None,
):
    pipeline = _build_pipeline(Path(rus_cefr_path), translator, aligner)
    return pipeline.predict(kaz_text, russian_text=russian_text)


def predict_text_cefr_ensemble(
    kaz_text: str,
    *,
    rus_cefr_path: str | Path = Path("data/cefr/russian_cefr_sample.csv"),
    translator: Translator | None = None,
    aligner: EmbeddingAligner | None = None,
    russian_text: str | None = None,
    russian_model_checkpoint: str | Path | None = None,
    russian_model: RuSentenceCefrModel | None = None,
    russian_weight: float = 0.6,
) -> dict[str, object]:
    pipeline = _build_pipeline(Path(rus_cefr_path), translator, aligner)
    if russian_model is None:
        if russian_model_checkpoint is None:
            raise ValueError("Either russian_model or russian_model_checkpoint must be provided.")
        russian_model = RuSentenceCefrModel.from_pretrained(russian_model_checkpoint)
    ensemble = EnsemblePipeline(
        base_pipeline=pipeline,
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
