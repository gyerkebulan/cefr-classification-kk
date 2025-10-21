from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import pymorphy3 as pymorphy
except ImportError:  # pragma: no cover
    import pymorphy2 as pymorphy

from cefr.alignment import is_informative
from cefr.data import CEFR_LEVELS

LEVEL_TO_INDEX: Mapping[str, int] = {level: idx for idx, level in enumerate(CEFR_LEVELS)}


@dataclass(slots=True, frozen=True)
class WeakLabelSample:
    sentence: str
    probabilities: tuple[float, float, float, float, float, float]


def load_russian_sentences(csv_path: str | Path, column: str = "rus") -> list[str]:
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is not present in {csv_path}.")
    sentences = df[column].astype(str).str.strip()
    return [sent for sent in sentences if sent]


def load_cefr_lexicon(csv_path: str | Path) -> Mapping[str, str]:
    df = pd.read_csv(csv_path)
    if not {"word", "level"}.issubset(df.columns):
        raise ValueError("Lexicon CSV must contain 'word' and 'level' columns.")
    mapping = {
        str(row.word).strip().lower(): str(row.level).strip()
        for row in df.itertuples(index=False)
        if str(row.word).strip()
    }
    return mapping


def _lemmatize_tokens(tokens: Sequence[str], analyzer: pymorphy.MorphAnalyzer) -> Iterable[str]:
    for token in tokens:
        token = token.strip().lower()
        if not is_informative(token):
            continue
        parsed = analyzer.parse(token)
        if not parsed:
            continue
        yield parsed[0].normal_form


def _sentence_probabilities(
    sentence: str, lexicon: Mapping[str, str], analyzer: pymorphy.MorphAnalyzer
) -> tuple[float, float, float, float, float, float]:
    tokens = sentence.split()
    counts = np.zeros(len(CEFR_LEVELS), dtype=np.float32)
    for lemma in _lemmatize_tokens(tokens, analyzer):
        level = lexicon.get(lemma)
        if level is None:
            continue
        idx = LEVEL_TO_INDEX.get(level)
        if idx is None:
            continue
        counts[idx] += 1.0
    total = counts.sum()
    if total == 0:
        return tuple(float(1.0 / len(CEFR_LEVELS)) for _ in CEFR_LEVELS)
    probs = counts / total
    return tuple(float(p) for p in probs)


def build_weak_labels(
    sentences: Sequence[str],
    lexicon: Mapping[str, str],
    analyzer: pymorphy.MorphAnalyzer | None = None,
) -> list[WeakLabelSample]:
    analyzer = analyzer or pymorphy.MorphAnalyzer()
    samples: list[WeakLabelSample] = []
    for sentence in sentences:
        probs = _sentence_probabilities(sentence, lexicon, analyzer)
        samples.append(WeakLabelSample(sentence=sentence, probabilities=probs))
    return samples


def save_weak_labels(
    samples: Sequence[WeakLabelSample],
    output_csv: str | Path,
) -> Path:
    output_csv = Path(output_csv)
    df = pd.DataFrame(
        [
            (sample.sentence, *sample.probabilities)
            for sample in samples
        ],
        columns=("sentence", *CEFR_LEVELS),
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    return output_csv


__all__ = [
    "WeakLabelSample",
    "CEFR_LEVELS",
    "load_russian_sentences",
    "load_cefr_lexicon",
    "build_weak_labels",
    "save_weak_labels",
]
