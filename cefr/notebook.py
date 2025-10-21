from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from cefr.config import NotebookConfig
from cefr.pipeline import EnsemblePipeline, EnsemblePrediction, TextPipeline
from cefr.models import RuSentenceCefrModel


AlignmentRow = tuple[int, int, str, str, str]


@dataclass(slots=True)
class NotebookPrediction:
    rows: list[AlignmentRow]
    cefr_level: str
    translation: str
    kazakh_distribution: dict[str, float]
    russian_distribution: dict[str, float] | None
    raw: object


def _rows_from_alignments(alignment_tuples: Iterable[tuple[str, str, int, int, str]]) -> list[AlignmentRow]:
    return [
        (kaz_idx, rus_idx, kaz_word, rus_word, cefr_level)
        for kaz_word, rus_word, kaz_idx, rus_idx, cefr_level in alignment_tuples
    ]


def predict_notebook_view(
    kazakh_text: str,
    *,
    config: NotebookConfig | None = None,
    russian_text: str | None = None,
    russian_model: RuSentenceCefrModel | None = None,
) -> NotebookPrediction:
    cfg = config or NotebookConfig()
    base_pipeline = TextPipeline(config=cfg.pipeline)

    if cfg.use_ensemble:
        if russian_model is None:
            checkpoint = cfg.pipeline.russian_model_dir
            if checkpoint is None:
                raise ValueError("Notebook config requested ensemble but no Russian model directory supplied.")
            russian_model = RuSentenceCefrModel.from_pretrained(checkpoint)
        ensemble = EnsemblePipeline(
            base_pipeline=base_pipeline,
            russian_model=russian_model,
            russian_weight=cfg.pipeline.russian_weight,
        )
        prediction: EnsemblePrediction = ensemble.predict(
            kazakh_text,
            russian_text=russian_text,
        )
        rows = _rows_from_alignments(prediction.word_alignments)
        return NotebookPrediction(
            rows=rows,
            cefr_level=prediction.level,
            translation=prediction.translation,
            kazakh_distribution=dict(prediction.kazakh_distribution),
            russian_distribution=dict(prediction.russian_distribution),
            raw=prediction,
        )

    base_prediction = base_pipeline.predict(kazakh_text, russian_text=russian_text)
    rows = _rows_from_alignments(base_prediction.word_alignment_tuples)
    return NotebookPrediction(
        rows=rows,
        cefr_level=base_prediction.average_level,
        translation=base_prediction.translation,
        kazakh_distribution=dict(base_prediction.distribution),
        russian_distribution=None,
        raw=base_prediction,
    )


def rows_to_dataframe(rows: Sequence[AlignmentRow]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "id_kazakh_word",
            "id_russian_word",
            "kazakh_word",
            "russian_word",
            "cefr_level",
        ],
    )


__all__ = ["NotebookPrediction", "predict_notebook_view", "rows_to_dataframe", "AlignmentRow"]
