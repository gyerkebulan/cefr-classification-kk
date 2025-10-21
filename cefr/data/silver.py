from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import pymorphy3 as pymorphy
except ImportError:  # pragma: no cover
    import pymorphy2 as pymorphy

from cefr.alignment import EmbeddingAligner, is_informative, merge_kz_to_single_ru
from cefr.config import AlignmentConfig
from cefr.data import RussianCefrRepository


def _lemmatizer() -> pymorphy.MorphAnalyzer:
    return pymorphy.MorphAnalyzer()


def _lemmatize(token: str, analyzer: pymorphy.MorphAnalyzer) -> str:
    parsed = analyzer.parse(token)[0]
    return parsed.normal_form


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(part for part in text.split() if part)


def build_silver_labels(
    parallel_csv: str | Path,
    *,
    rus_cefr: str | Path,
    out_csv: str | Path,
    aligner: EmbeddingAligner | None = None,
    alignment_config: AlignmentConfig | None = None,
    skip_non_informative: bool = True,
) -> Path:
    parallel_csv = Path(parallel_csv)
    rus_cefr = Path(rus_cefr)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(parallel_csv)
    repo = RussianCefrRepository(rus_cefr)
    mapping = repo.mapping

    if aligner is None:
        aligner = EmbeddingAligner(alignment_config)
    analyzer = _lemmatizer()

    rows: list[dict[str, str]] = []
    skipped_sequences = 0

    for sample in df.itertuples(index=False):
        kazakh = str(getattr(sample, "kaz", "")).strip()
        russian = str(getattr(sample, "rus", "")).strip()
        kz_words = _tokenize(kazakh)
        ru_words = _tokenize(russian)
        try:
            links = aligner.align(kz_words, ru_words)
        except Exception:
            skipped_sequences += 1
            continue
        phrases = merge_kz_to_single_ru(kz_words, ru_words, links)
        if not phrases:
            skipped_sequences += 1
            continue

        for phrase in phrases:
            token = phrase.russian_token.strip().lower()
            if skip_non_informative and not is_informative(token):
                continue
            lemma = _lemmatize(token, analyzer)
            level = mapping.get(lemma, mapping.get(token, "Unknown"))
            rows.append(
                {
                    "kaz_item": phrase.kazakh_phrase,
                    "rus_item": phrase.russian_token,
                    "cefr": level,
                    "kaz_sent": kazakh,
                    "rus_sent": russian,
                }
            )

    output_df = pd.DataFrame(rows, columns=["kaz_item", "rus_item", "cefr", "kaz_sent", "rus_sent"])
    output_df.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"Saved: {out_csv} rows={len(rows)} skipped_sentences={skipped_sequences}")
    return out_csv


__all__ = ["build_silver_labels"]
