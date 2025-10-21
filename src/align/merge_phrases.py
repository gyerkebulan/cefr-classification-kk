from __future__ import annotations

from collections import defaultdict
from typing import Sequence, Set

from src.domain.entities import PhraseAlignment


def merge_kz_to_single_ru(
    kazakh_words: Sequence[str],
    russian_words: Sequence[str],
    links: Set[tuple[int, int]],
) -> list[PhraseAlignment]:
    """Merge contiguous Kazakh token spans that align to a single Russian token."""

    alignment_index = defaultdict(list)
    for kazakh_idx, russian_idx in links:
        alignment_index[russian_idx].append(kazakh_idx)

    merged_spans: list[tuple[tuple[int, ...], int]] = []
    for russian_idx in sorted(alignment_index):
        kazakh_indices = sorted(alignment_index[russian_idx])
        span = [kazakh_indices[0]]
        for idx in kazakh_indices[1:]:
            if idx == span[-1] + 1:
                span.append(idx)
            else:
                merged_spans.append((tuple(span), russian_idx))
                span = [idx]
        merged_spans.append((tuple(span), russian_idx))

    results: list[PhraseAlignment] = []
    for kazakh_span, russian_idx in merged_spans:
        kazakh_phrase = " ".join(kazakh_words[i] for i in kazakh_span)
        russian_token = russian_words[russian_idx]
        results.append(
            PhraseAlignment(
                kazakh_phrase=kazakh_phrase,
                russian_token=russian_token,
                kazakh_span=kazakh_span,
                russian_index=russian_idx,
            )
        )
    return results
