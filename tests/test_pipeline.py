from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Set, Tuple

from src.align.merge_phrases import merge_kz_to_single_ru
from src.text.predict_text import predict_text_cefr


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
