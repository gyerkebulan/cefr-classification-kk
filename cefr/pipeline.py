from pathlib import Path

from cefr.alignment import (
    EmbeddingAligner,
    WordAlignment,
    merge_kz_to_single_ru,
)
from cefr.config import PipelineConfig
from cefr.data import RussianCefrRepository
from cefr.models import RuSentenceCefrModel
from cefr.scoring import CefrScorer
from cefr.translation import get_translator
from cefr.text_utils import tokenize_words


class TextPrediction:
    __slots__ = ("translation", "distribution", "average_level", "phrase_alignments", "word_alignments")

    def __init__(self, translation, distribution, average_level, phrase_alignments, word_alignments):
        self.translation = translation
        self.distribution = distribution
        self.average_level = average_level
        self.phrase_alignments = phrase_alignments
        self.word_alignments = word_alignments

    def iter_word_alignment_tuples(self):
        for alignment in self.word_alignments:
            yield alignment.as_tuple()

    @property
    def word_alignment_tuples(self):
        return list(self.iter_word_alignment_tuples())

    def find_word_alignment(self, word, *, case_insensitive=True):
        if not self.word_alignments:
            return None
        target = word.lower() if case_insensitive else word
        for alignment in self.word_alignments:
            token = alignment.kazakh_token
            token_cmp = token.lower() if case_insensitive else token
            if token_cmp == target:
                return alignment
        return None

    def level_for_word(self, word, *, case_insensitive=True):
        alignment = self.find_word_alignment(word, case_insensitive=case_insensitive)
        if alignment is None:
            return None
        level = alignment.cefr
        confidence = float(self.distribution.get(level, 0.0))
        return level, confidence, alignment


class TextPipeline:
    """Translate, align and score Kazakh text."""

    def __init__(
        self,
        *,
        config=None,
        translator=None,
        aligner=None,
        scorer=None,
        repository=None,
    ):
        self.config = config or PipelineConfig()
        self.translator = translator or get_translator(self.config.translator)
        aligner_config = self.config.alignment
        self.aligner = aligner or EmbeddingAligner(aligner_config)
        repo_path = repository.path if repository else self.config.russian_cefr_path
        self.repository = repository or RussianCefrRepository(Path(repo_path))
        self.scorer = scorer or CefrScorer(self.repository)

    def predict(self, kazakh_text, *, russian_text=None):
        translation = russian_text or self.translator.translate(kazakh_text)
        kazakh_words = tokenize_words(kazakh_text)
        russian_words = tokenize_words(translation)
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
        self, phrases, kazakh_words
    ):
        word_alignments = []
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


class EnsemblePrediction:
    __slots__ = (
        "level",
        "confidence",
        "probabilities",
        "translation",
        "kazakh_distribution",
        "russian_distribution",
        "word_alignments",
        "base_prediction",
    )

    def __init__(
        self,
        level,
        confidence,
        probabilities,
        translation,
        kazakh_distribution,
        russian_distribution,
        word_alignments,
        base_prediction,
    ):
        self.level = level
        self.confidence = confidence
        self.probabilities = probabilities
        self.translation = translation
        self.kazakh_distribution = kazakh_distribution
        self.russian_distribution = russian_distribution
        self.word_alignments = word_alignments
        self.base_prediction = base_prediction


class EnsemblePipeline:
    def __init__(
        self,
        *,
        base_pipeline,
        russian_model,
        russian_weight=0.6,
    ):
        if not 0.0 <= russian_weight <= 1.0:
            raise ValueError("russian_weight must be between 0 and 1 inclusive.")
        self.base = base_pipeline
        self.russian_model = russian_model
        self.weight = russian_weight

    def _weighted_average(
        self, ru_probs, kk_probs
    ):
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

    def predict(self, kazakh_text, *, russian_text=None):
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
