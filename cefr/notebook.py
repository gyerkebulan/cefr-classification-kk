import pandas as pd

from cefr.config import NotebookConfig
from cefr.pipeline import EnsemblePipeline, TextPipeline
from cefr.models import RuSentenceCefrModel


class NotebookPrediction:
    __slots__ = ("rows", "cefr_level", "translation", "kazakh_distribution", "russian_distribution", "raw")

    def __init__(self, rows, cefr_level, translation, kazakh_distribution, russian_distribution, raw):
        self.rows = rows
        self.cefr_level = cefr_level
        self.translation = translation
        self.kazakh_distribution = kazakh_distribution
        self.russian_distribution = russian_distribution
        self.raw = raw


def _rows_from_alignments(alignment_tuples):
    return [
        (kaz_idx, rus_idx, kaz_word, rus_word, cefr_level)
        for kaz_word, rus_word, kaz_idx, rus_idx, cefr_level in alignment_tuples
    ]


def predict_notebook_view(
    kazakh_text,
    *,
    config=None,
    russian_text=None,
    russian_model=None,
):
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
        prediction = ensemble.predict(
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


def rows_to_dataframe(rows):
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


__all__ = ["NotebookPrediction", "predict_notebook_view", "rows_to_dataframe"]
