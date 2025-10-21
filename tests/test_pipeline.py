from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Set, Tuple

from cefr.alignment import WordAlignment, merge_kz_to_single_ru
from cefr.config import PipelineConfig
from cefr.data import CEFR_LEVELS
from cefr.pipeline import EnsemblePipeline, TextPipeline, TextPrediction


@dataclass
class DummyTranslator:
    expected_text: str
    called_with: list[str]

    def translate(self, text: str) -> str:
        self.called_with.append(text)
        return self.expected_text


class DummyAligner:
    def align(
        self,
        kazakh_words: Sequence[str],
        russian_words: Sequence[str],
        *,
        layer: int | None = None,
        threshold: float | None = None,
    ) -> Set[Tuple[int, int]]:
        if not kazakh_words or not russian_words:
            return set()
        return {(0, 0)}


def test_merge_kz_to_single_ru_simple():
    kz_words = ["Ол", "кітап", "оқып", "жатыр"]
    ru_words = ["Он", "читает", "книгу"]
    links = {(0, 0), (1, 1), (2, 1), (3, 2)}

    phrases = merge_kz_to_single_ru(kz_words, ru_words, links)

    assert len(phrases) == 3
    assert phrases[0].kazakh_phrase == "Ол"
    assert phrases[0].russian_token == "Он"
    assert phrases[1].kazakh_phrase == "кітап оқып"
    assert phrases[1].russian_token == "читает"


def test_pipeline_respects_translation_override(tmp_path, monkeypatch):
    dummy_translator = DummyTranslator(expected_text="модель", called_with=[])
    dummy_aligner = DummyAligner()

    # Ensure repository path exists by pointing to bundled CSV.
    config = PipelineConfig()
    pipeline = TextPipeline(
        config=config,
        translator=dummy_translator,  # type: ignore[arg-type]
        aligner=dummy_aligner,  # type: ignore[arg-type]
    )

    result = pipeline.predict("Ол кітап оқып жатыр", russian_text="человек")

    assert dummy_translator.called_with == []
    assert result.translation == "человек"
    assert result.phrase_alignments
    assert isinstance(result, TextPrediction)
    assert result.average_level in {"A1", "Unknown"}
    assert all(isinstance(item, WordAlignment) for item in result.word_alignments)


class DummyRuSentenceModel:
    def __init__(self, mapping: Mapping[str, float]):
        self._mapping = mapping

    def predict_proba(self, sentence: str) -> Mapping[str, float]:
        return self._mapping


class DummyKazPipeline:
    def __init__(self) -> None:
        word_alignment = WordAlignment(
            kazakh_token="Ол",
            russian_token="Он",
            kazakh_index=0,
            russian_index=0,
            cefr="A1",
        )
        self.prediction = TextPrediction(
            translation="Он",
            distribution={level: (1.0 if level == "A1" else 0.0) for level in CEFR_LEVELS},
            average_level="A1",
            phrase_alignments=(),
            word_alignments=(word_alignment,),
        )
        self.scorer = type("Scorer", (), {"cefr_order": CEFR_LEVELS})()

    def predict(self, kazakh_text: str, *, russian_text: str | None = None) -> TextPrediction:
        return self.prediction


def test_ensemble_pipeline_combines_predictions():
    kk_pipeline = DummyKazPipeline()
    ru_probs = {level: 1.0 / len(CEFR_LEVELS) for level in CEFR_LEVELS}
    ru_model = DummyRuSentenceModel(ru_probs)

    ensemble = EnsemblePipeline(
        base_pipeline=kk_pipeline,  # type: ignore[arg-type]
        russian_model=ru_model,  # type: ignore[arg-type]
        russian_weight=0.6,
    )

    prediction = ensemble.predict("Ол кітап оқып жатыр", russian_text="человек")

    assert set(prediction.probabilities.keys()) == set(CEFR_LEVELS)
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-6
    assert prediction.translation == "Он"
    assert prediction.word_alignments
