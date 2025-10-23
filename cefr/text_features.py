import math
import re
from dataclasses import asdict, dataclass

from cefr.text_utils import tokenize_words

__all__ = ["TextFeatures", "compute_text_features", "extract_numbers", "split_sentences"]
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.MULTILINE)
VOWELS = "aeiouyаеёиоуыэюя"


@dataclass(slots=True)
class TextFeatures:
    sentence_count: int
    token_count: int
    type_count: int
    average_sentence_length: float
    type_token_ratio: float
    number_count: int
    syllable_count: int
    average_syllables_per_word: float
    long_word_percentage: float

    def as_dict(self):
        return asdict(self)


def split_sentences(text):
    candidates = [match.group().strip() for match in SENTENCE_PATTERN.finditer(text)]
    return [candidate for candidate in candidates if candidate]


def extract_numbers(text):
    return [number for number in NUMBER_PATTERN.findall(text)]


def count_syllables(word):
    word_lower = word.lower()
    if not word_lower:
        return 0
    matches = re.findall(rf"[{VOWELS}]+", word_lower)
    syllable_groups = len(matches)
    if syllable_groups == 0:
        return 1
    if word_lower.endswith(("e", "es")) and syllable_groups > 1:
        syllable_groups -= 1
    return max(1, syllable_groups)


def iter_syllable_counts(tokens):
    for token in tokens:
        yield count_syllables(token)


def compute_text_features(text):
    sentences = split_sentences(text)
    words = tokenize_words(text)
    numbers = set(extract_numbers(text))

    words_excluding_numbers = [word for word in words if word not in numbers]
    tokens_total = len(words_excluding_numbers)
    unique_tokens = {word.lower() for word in words_excluding_numbers}
    unique_total = len(unique_tokens)
    number_count = len(numbers)
    sentence_count = len(sentences)

    syllable_counts = list(iter_syllable_counts(words_excluding_numbers))
    total_syllables = sum(syllable_counts)
    long_words = sum(1 for count in syllable_counts if count > 2)

    average_sentence_length = (
        float(tokens_total) / float(sentence_count) if sentence_count > 0 else 0.0
    )
    average_syllables = float(total_syllables) / float(tokens_total) if tokens_total > 0 else 0.0
    type_token_ratio = (
        float(unique_total) / float(tokens_total) if tokens_total > 0 else 0.0
    )
    long_word_percentage = (
        (float(long_words) / float(tokens_total) * 100.0) if tokens_total > 0 else 0.0
    )

    if math.isfinite(average_sentence_length) is False:
        average_sentence_length = 0.0
    if math.isfinite(average_syllables) is False:
        average_syllables = 0.0
    if math.isfinite(type_token_ratio) is False:
        type_token_ratio = 0.0
    if math.isfinite(long_word_percentage) is False:
        long_word_percentage = 0.0

    return TextFeatures(
        sentence_count=sentence_count,
        token_count=tokens_total,
        type_count=unique_total,
        average_sentence_length=average_sentence_length,
        type_token_ratio=type_token_ratio,
        number_count=number_count,
        syllable_count=total_syllables,
        average_syllables_per_word=average_syllables,
        long_word_percentage=long_word_percentage,
    )
