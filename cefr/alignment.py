from __future__ import annotations

from collections import defaultdict
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Sequence, Tuple

import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .config import AlignmentConfig


class SequenceTooLongError(RuntimeError):
    """Raised when tokenized sequences exceed the alignment model limits."""


@dataclass(slots=True)
class PhraseAlignment:
    kazakh_phrase: str
    russian_token: str
    kazakh_span: tuple[int, ...]
    russian_index: int


@dataclass(slots=True)
class WordAlignment:
    kazakh_token: str
    russian_token: str
    kazakh_index: int
    russian_index: int
    cefr: str

    def as_tuple(self) -> tuple[str, str, int, int, str]:
        return (
            self.kazakh_token,
            self.russian_token,
            self.kazakh_index,
            self.russian_index,
            self.cefr,
        )


@dataclass(slots=True)
class AlignmentResources:
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    device: torch.device


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@lru_cache(maxsize=4)
def _load_resources(model_name: str, device_hint: str | None) -> AlignmentResources:
    device = _resolve_device(device_hint)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return AlignmentResources(tokenizer=tokenizer, model=model, device=device)


class EmbeddingAligner:
    """Mutual alignment over contextual embeddings."""

    def __init__(self, config: AlignmentConfig | None = None) -> None:
        self.config = config or AlignmentConfig()
        resources = _load_resources(self.config.model_name, self.config.device)
        self.tokenizer = resources.tokenizer
        self.model = resources.model
        self.device = resources.device
        max_len = getattr(self.tokenizer, "model_max_length", 512)
        if max_len is None or max_len <= 0 or max_len > 512:
            max_len = 512
        self.max_length = int(max_len)

    def align(
        self,
        kz_words: Sequence[str],
        ru_words: Sequence[str],
        *,
        layer: int | None = None,
        threshold: float | None = None,
    ) -> set[tuple[int, int]]:
        layer = layer if layer is not None else self.config.layer
        threshold = threshold if threshold is not None else self.config.threshold
        kz_enc, kz_ids, kz_truncated = self._tokenize_words(kz_words)
        ru_enc, ru_ids, ru_truncated = self._tokenize_words(ru_words)
        if kz_truncated or ru_truncated:
            raise SequenceTooLongError("Sequence exceeds maximum length for alignment model.")

        kz_hs = self._hidden_state(kz_enc, layer)
        ru_hs = self._hidden_state(ru_enc, layer)
        kz_rep, kz_keep = self._pool_words(kz_hs, kz_ids)
        ru_rep, ru_keep = self._pool_words(ru_hs, ru_ids)
        if kz_rep.numel() == 0 or ru_rep.numel() == 0:
            return set()
        sim = kz_rep @ ru_rep.T
        p_rgk = torch.softmax(sim, dim=-1)
        p_kgr = torch.softmax(sim, dim=-2)
        links: set[tuple[int, int]] = set()
        for i, kz_idx in enumerate(kz_keep):
            for j, ru_idx in enumerate(ru_keep):
                if p_rgk[i, j] > threshold and p_kgr[i, j] > threshold:
                    links.add((kz_idx, ru_idx))
        return links

    def diagnostics(
        self,
        kz_words: Sequence[str],
        ru_words: Sequence[str],
        *,
        layer: int | None = None,
        threshold: float | None = None,
    ) -> "AlignmentDiagnostics":
        layer = layer if layer is not None else self.config.layer
        threshold = threshold if threshold is not None else self.config.threshold
        kz_enc, kz_ids, kz_truncated = self._tokenize_words(kz_words)
        ru_enc, ru_ids, ru_truncated = self._tokenize_words(ru_words)
        if kz_truncated or ru_truncated:
            raise SequenceTooLongError("Sequence exceeds maximum length for alignment model.")
        kz_hs = self._hidden_state(kz_enc, layer)
        ru_hs = self._hidden_state(ru_enc, layer)
        kz_rep, kz_keep = self._pool_words(kz_hs, kz_ids)
        ru_rep, ru_keep = self._pool_words(ru_hs, ru_ids)
        if kz_rep.numel() == 0 or ru_rep.numel() == 0:
            empty = torch.empty((0, 0), device=self.device)
            return AlignmentDiagnostics(set(), [], [], empty, empty)
        sim = kz_rep @ ru_rep.T
        p_rgk = torch.softmax(sim, dim=-1)
        p_kgr = torch.softmax(sim, dim=-2)
        links = {
            (kz_keep[i], ru_keep[j])
            for i in range(p_rgk.size(0))
            for j in range(p_rgk.size(1))
            if p_rgk[i, j] > threshold and p_kgr[i, j] > threshold
        }
        return AlignmentDiagnostics(links, list(kz_keep), list(ru_keep), p_rgk, p_kgr)

    def _tokenize_words(self, words: Sequence[str]):
        enc = self.tokenizer(
            list(words),
            is_split_into_words=True,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
            max_length=self.max_length,
            padding="longest",
        )
        word_ids = enc.word_ids(0)
        max_word_id = max((wid for wid in word_ids if wid is not None), default=-1)
        truncated = max_word_id < len(words) - 1
        for key, value in list(enc.items()):
            if isinstance(value, torch.Tensor):
                enc[key] = value.to(self.device)
        return enc, word_ids, truncated

    def _hidden_state(self, enc, layer: int):
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        return out.hidden_states[layer].squeeze(0)

    @staticmethod
    def _pool_words(hs, word_ids: Iterable[int | None]):
        buckets: dict[int, list[torch.Tensor]] = defaultdict(list)
        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            buckets[wid].append(hs[i])
        keep = sorted(buckets.keys())
        if not keep:
            return torch.empty((0, hs.size(-1)), device=hs.device), keep
        reps = torch.stack([torch.stack(buckets[k]).mean(0) for k in keep])
        return reps, keep


@dataclass(slots=True)
class AlignmentDiagnostics:
    links: set[tuple[int, int]]
    kz_keep: list[int]
    ru_keep: list[int]
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
        return float(min(self.p_ru_given_kz[i, j], self.p_kz_given_ru[i, j]))


def merge_kz_to_single_ru(
    kazakh_words: Sequence[str],
    russian_words: Sequence[str],
    links: set[tuple[int, int]],
) -> list[PhraseAlignment]:
    alignment_index: dict[int, list[int]] = defaultdict(list)
    for kaz_idx, ru_idx in links:
        alignment_index[ru_idx].append(kaz_idx)

    merged_spans: list[tuple[tuple[int, ...], int]] = []
    for ru_idx in sorted(alignment_index):
        kazakh_indices = sorted(alignment_index[ru_idx])
        span = [kazakh_indices[0]]
        for idx in kazakh_indices[1:]:
            if idx == span[-1] + 1:
                span.append(idx)
            else:
                merged_spans.append((tuple(span), ru_idx))
                span = [idx]
        merged_spans.append((tuple(span), ru_idx))

    result: list[PhraseAlignment] = []
    for kazakh_span, ru_idx in merged_spans:
        kazakh_phrase = " ".join(kazakh_words[i] for i in kazakh_span)
        russian_token = russian_words[ru_idx]
        result.append(
            PhraseAlignment(
                kazakh_phrase=kazakh_phrase,
                russian_token=russian_token,
                kazakh_span=kazakh_span,
                russian_index=ru_idx,
            )
        )
    return result


NON_WORD_RE = re.compile(r"^(?:\W+|\d[\d\W]*)$") # TODO: Подкорректировать regex для чисто русских/казахских слов


def is_informative(token: str) -> bool:
    token = token.strip().lower()
    return bool(token) and not NON_WORD_RE.fullmatch(token)


def informative_link_share(
    details: AlignmentDiagnostics,
    kz_words: Sequence[str],
    ru_words: Sequence[str],
    *,
    gold_links: set[tuple[int, int]] | None = None,
) -> float:
    if not details.links:
        return 0.0

    candidate = details.links if gold_links is None else details.links & gold_links
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
        details = aligner.diagnostics(kz_words, ru_words, layer=layer, threshold=thresh)
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
    "EmbeddingAligner",
    "AlignmentDiagnostics",
    "SequenceTooLongError",
    "PhraseAlignment",
    "WordAlignment",
    "merge_kz_to_single_ru",
    "is_informative",
    "informative_link_share",
    "fraction_above_threshold",
]
