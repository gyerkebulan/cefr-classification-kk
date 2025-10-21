# src/align/mutual_align.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence, Set, Tuple

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# Use the awesome-align checkpoint (mBERT fine-tuned for alignment)
MODEL_NAME = "aneuraz/awesome-align-with-co"  # or "bert-base-multilingual-cased" as a fallback


@dataclass(frozen=True)
class AlignmentResources:
    """Container for resources required to run mutual alignment."""

    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    device: torch.device


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@lru_cache(maxsize=4)
def _load_alignment_resources(
    model_name: str = MODEL_NAME, device: str | torch.device | None = None
) -> AlignmentResources:
    resolved_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(resolved_device)
    return AlignmentResources(tokenizer=tokenizer, model=model, device=resolved_device)


class SequenceTooLongError(RuntimeError):
    """Raised when tokenized sequences exceed the model maximum length."""


class EmbeddingAligner:
    """Mutual soft alignment over contextual embeddings with optional caching."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | torch.device | None = None,
        resources: AlignmentResources | None = None,
    ) -> None:
        if resources is None:
            resources = _load_alignment_resources(model_name=model_name, device=device)
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
        layer: int = 8,
        thresh: float = 0.05,
    ) -> Set[Tuple[int, int]]:
        kz_enc, kz_wids, kz_truncated = self._tokenize_words(kz_words)
        ru_enc, ru_wids, ru_truncated = self._tokenize_words(ru_words)
        if kz_truncated or ru_truncated:
            raise SequenceTooLongError(
                f"Sequence exceeds maximum length ({self.max_length}) "
                f"kz_truncated={kz_truncated} ru_truncated={ru_truncated}"
            )
        kz_hs = self._layer_hs(kz_enc, layer)
        ru_hs = self._layer_hs(ru_enc, layer)
        kz_rep, kz_keep = self._pool_words(kz_hs, kz_wids)
        ru_rep, ru_keep = self._pool_words(ru_hs, ru_wids)
        if kz_rep.numel() == 0 or ru_rep.numel() == 0:
            return set()
        sim = kz_rep @ ru_rep.T
        p_rgk = torch.softmax(sim, -1)
        p_kgr = torch.softmax(sim, -2)
        links: Set[Tuple[int, int]] = set()
        for i in range(p_rgk.size(0)):
            for j in range(p_rgk.size(1)):
                if p_rgk[i, j] > thresh and p_kgr[i, j] > thresh:
                    links.add((kz_keep[i], ru_keep[j]))
        return links

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

    def _layer_hs(self, enc, layer: int = 8):
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        return out.hidden_states[layer].squeeze(0)  # [T,H]

    @staticmethod
    def _pool_words(hs, word_ids: Iterable[int | None]):
        buckets = defaultdict(list)
        for i, wid in enumerate(word_ids):
            if wid is None:  # specials
                continue
            buckets[wid].append(hs[i])
        keep = sorted(buckets.keys())
        if not keep:
            return torch.empty((0, hs.size(-1)), device=hs.device), keep
        reps = torch.stack([torch.stack(buckets[k]).mean(0) for k in keep])
        return reps, keep


_default_aligner: EmbeddingAligner | None = None


def get_default_aligner() -> EmbeddingAligner:
    global _default_aligner
    if _default_aligner is None:
        _default_aligner = EmbeddingAligner()
    return _default_aligner


def reset_default_aligner() -> None:
    """Reset the cached aligner (mainly for tests)."""

    global _default_aligner
    _default_aligner = None


def mutual_soft_align(
    kz_words: Sequence[str],
    ru_words: Sequence[str],
    layer: int = 8,
    thresh: float = 0.05,
    aligner: EmbeddingAligner | None = None,
) -> Set[Tuple[int, int]]:
    """Backward-compatible helper around :class:`EmbeddingAligner`."""

    if aligner is None:
        aligner = get_default_aligner()
    return aligner.align(kz_words, ru_words, layer=layer, thresh=thresh)


__all__ = [
    "SequenceTooLongError",
    "EmbeddingAligner",
    "mutual_soft_align",
    "get_default_aligner",
    "reset_default_aligner",
]
