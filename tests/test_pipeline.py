from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Set, Tuple

from src.align.merge_phrases import merge_kz_to_single_ru
from src.domain.entities import WordAlignment, TextCefrPrediction
from src.text.ensemble_pipeline import EnsembleCefrPipeline
from src.text.predict_text import predict_text_cefr
from src.ru_sentence_model.data import CEFR_LEVELS


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
        layer: int = 8,
        thresh: float = 0.05,
    ) -> Set[Tuple[int, int]]:
        # Always align the first token in each sequence.
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


def test_predict_text_cefr_uses_override_translation():
    dummy_translator = DummyTranslator(expected_text="модель", called_with=[])
    dummy_aligner = DummyAligner()

    result = predict_text_cefr(
        "Ол кітап оқып жатыр",
        translator=dummy_translator,
        aligner=dummy_aligner,
        russian_text="человек",
    )

    # Translation override should bypass translator invocation.
    assert dummy_translator.called_with == []
    assert result.translation == "человек"
    assert result.phrase_alignments  # alignment ran with dummy aligner
    assert result.average_level in {"A1", "Unknown"}
    assert all(isinstance(item, WordAlignment) for item in result.word_alignments)
    tuples = [item.as_tuple() for item in result.word_alignments]
    assert tuples
    for entry in tuples:
        assert len(entry) == 5


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
        self.prediction = TextCefrPrediction(
            translation="Он",
            distribution={level: (1.0 if level == "A1" else 0.0) for level in CEFR_LEVELS},
            average_level="A1",
            phrase_alignments=(),
            word_alignments=(word_alignment,),
        )

    def predict(self, kazakh_text: str, *, russian_text: str | None = None) -> TextCefrPrediction:
        return self.prediction


def test_ensemble_pipeline_combines_predictions():
    kk_pipeline = DummyKazPipeline()
    ru_probs = {level: 1.0 / len(CEFR_LEVELS) for level in CEFR_LEVELS}
    ru_model = DummyRuSentenceModel(ru_probs)

    ensemble = EnsembleCefrPipeline(
        kazakh_pipeline=kk_pipeline,
        russian_model=ru_model,  # type: ignore[arg-type]
        russian_weight=0.6,
    )

    prediction = ensemble.predict("Ол кітап оқып жатыр", russian_text="человек")

    assert set(prediction.probabilities.keys()) == set(CEFR_LEVELS)
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-6
    assert prediction.translation == "Он"
    assert prediction.word_alignments
