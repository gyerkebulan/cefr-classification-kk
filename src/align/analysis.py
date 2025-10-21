from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Set, Tuple

import torch
import pandas as pd

from .mutual_align import EmbeddingAligner, SequenceTooLongError

NON_WORD_RE = re.compile(r"^(?:\W+|\d[\d\W]*)$")


def is_informative(token: str) -> bool:
    token = token.strip().lower()
    return bool(token) and not NON_WORD_RE.fullmatch(token)


@dataclass(slots=True)
class AlignmentDetails:
    links: Set[Tuple[int, int]]
    kz_keep: Sequence[int]
    ru_keep: Sequence[int]
    p_ru_given_kz: torch.Tensor
    p_kz_given_ru: torch.Tensor

    def iter_rows(
        self, kz_words: Sequence[str], ru_words: Sequence[str]
    ) -> Iterable[Mapping[str, object]]:
        for i, kz_idx in enumerate(self.kz_keep):
            kz_token = kz_words[kz_idx]
            for j, ru_idx in enumerate(self.ru_keep):
                ru_token = ru_words[ru_idx]
                prob_rgk = float(self.p_ru_given_kz[i, j])
                prob_kgr = float(self.p_kz_given_ru[i, j])
                yield {
                    "kaz_index": kz_idx,
                    "kaz_token": kz_token,
                    "rus_index": ru_idx,
                    "rus_token": ru_token,
                    "p_ru_given_kz": prob_rgk,
                    "p_kz_given_ru": prob_kgr,
                    "joint_prob": float(min(prob_rgk, prob_kgr)),
                    "is_link": (kz_idx, ru_idx) in self.links,
                }

    def to_dataframe(self, kz_words: Sequence[str], ru_words: Sequence[str]) -> pd.DataFrame:
        rows = list(self.iter_rows(kz_words, ru_words))
        if not rows:
            return pd.DataFrame(
                columns=[
                    "kaz_index",
                    "kaz_token",
                    "rus_index",
                    "rus_token",
                    "p_ru_given_kz",
                    "p_kz_given_ru",
                    "joint_prob",
                    "is_link",
                ]
            )
        return pd.DataFrame(rows)

    def link_probability(self, kz_idx: int, ru_idx: int) -> float:
        kz_lookup = {idx: i for i, idx in enumerate(self.kz_keep)}
        ru_lookup = {idx: j for j, idx in enumerate(self.ru_keep)}
        i = kz_lookup[kz_idx]
        j = ru_lookup[ru_idx]
        return float(
            min(self.p_ru_given_kz[i, j], self.p_kz_given_ru[i, j])
        )


def align_with_probabilities(
    aligner: EmbeddingAligner,
    kz_words: Sequence[str],
    ru_words: Sequence[str],
    *,
    layer: int = 8,
    thresh: float = 0.05,
) -> AlignmentDetails:
    kz_enc, kz_wids, kz_truncated = aligner._tokenize_words(kz_words)
    ru_enc, ru_wids, ru_truncated = aligner._tokenize_words(ru_words)
    if kz_truncated or ru_truncated:
        raise SequenceTooLongError("Sequence exceeds maximum length for alignment model.")

    kz_hs = aligner._layer_hs(kz_enc, layer)
    ru_hs = aligner._layer_hs(ru_enc, layer)
    kz_rep, kz_keep = aligner._pool_words(kz_hs, kz_wids)
    ru_rep, ru_keep = aligner._pool_words(ru_hs, ru_wids)
    if kz_rep.numel() == 0 or ru_rep.numel() == 0:
        empty = torch.empty((0, 0), device=aligner.device)
        return AlignmentDetails(set(), [], [], empty, empty)

    sim = kz_rep @ ru_rep.T
    p_rgk = torch.softmax(sim, dim=-1)
    p_kgr = torch.softmax(sim, dim=-2)

    links: Set[Tuple[int, int]] = set()
    for i, kz_idx in enumerate(kz_keep):
        for j, ru_idx in enumerate(ru_keep):
            if p_rgk[i, j] > thresh and p_kgr[i, j] > thresh:
                links.add((kz_idx, ru_idx))

    return AlignmentDetails(links, list(kz_keep), list(ru_keep), p_rgk, p_kgr)


def informative_link_share(
    details: AlignmentDetails,
    kz_words: Sequence[str],
    ru_words: Sequence[str],
    *,
    gold_links: Set[Tuple[int, int]] | None = None,
) -> float:
    if not details.links:
        return 0.0

    if gold_links is None:
        candidate = details.links
    else:
        candidate = details.links & gold_links

    informative = {
        link
        for link in candidate
        if is_informative(kz_words[link[0]]) and is_informative(ru_words[link[1]])
    }
    return len(informative) / len(details.links)


def fraction_above_threshold(
    samples: Sequence[Mapping[str, str]],
    aligner: EmbeddingAligner,
    *,
    layer: int = 8,
    thresh: float = 0.05,
    prob_threshold: float = 0.2,
    kaz_key: str = "kaz_sent",
    rus_key: str = "rus_sent",
) -> float:
    total_links = 0
    passed = 0
    for item in samples:
        kz_words = tuple(item[kaz_key].split())
        ru_words = tuple(item[rus_key].split())
        details = align_with_probabilities(aligner, kz_words, ru_words, layer=layer, thresh=thresh)
        if not details.links:
            continue
        kz_lookup = {idx: i for i, idx in enumerate(details.kz_keep)}
        ru_lookup = {idx: j for j, idx in enumerate(details.ru_keep)}
        for link in details.links:
            kz_idx, ru_idx = link
            i = kz_lookup[kz_idx]
            j = ru_lookup[ru_idx]
            joint = float(min(details.p_ru_given_kz[i, j], details.p_kz_given_ru[i, j]))
            total_links += 1
            if joint >= prob_threshold:
                passed += 1
    return passed / total_links if total_links else 0.0


__all__ = [
    "AlignmentDetails",
    "align_with_probabilities",
    "fraction_above_threshold",
    "informative_link_share",
    "is_informative",
]
