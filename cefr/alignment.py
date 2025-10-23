from collections import defaultdict
import re
from functools import lru_cache
import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer

from .config import AlignmentConfig
from .text_utils import is_cyrillic_token


class SequenceTooLongError(RuntimeError):
    """Raised when tokenized sequences exceed the alignment model limits."""


class PhraseAlignment:
    __slots__ = ("kazakh_phrase", "russian_token", "kazakh_span", "russian_index")

    def __init__(self, kazakh_phrase, russian_token, kazakh_span, russian_index):
        self.kazakh_phrase = kazakh_phrase
        self.russian_token = russian_token
        self.kazakh_span = kazakh_span
        self.russian_index = russian_index


class WordAlignment:
    __slots__ = ("kazakh_token", "russian_token", "kazakh_index", "russian_index", "cefr")

    def __init__(self, kazakh_token, russian_token, kazakh_index, russian_index, cefr):
        self.kazakh_token = kazakh_token
        self.russian_token = russian_token
        self.kazakh_index = kazakh_index
        self.russian_index = russian_index
        self.cefr = cefr

    def as_tuple(self):
        return (
            self.kazakh_token,
            self.russian_token,
            self.kazakh_index,
            self.russian_index,
            self.cefr,
        )


class AlignmentResources:
    __slots__ = ("tokenizer", "model", "device")

    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device


def _resolve_device(device):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@lru_cache(maxsize=4)
def _load_resources(model_name, device_hint):
    device = _resolve_device(device_hint)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return AlignmentResources(tokenizer=tokenizer, model=model, device=device)


class EmbeddingAligner:
    """Mutual alignment over contextual embeddings."""

    def __init__(self, config=None):
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
        kz_words,
        ru_words,
        *,
        layer=None,
        threshold=None,
    ):
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
        links = set()
        for i, kz_idx in enumerate(kz_keep):
            for j, ru_idx in enumerate(ru_keep):
                if p_rgk[i, j] > threshold and p_kgr[i, j] > threshold:
                    links.add((kz_idx, ru_idx))
        return links

    def diagnostics(
        self,
        kz_words,
        ru_words,
        *,
        layer=None,
        threshold=None,
    ):
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

    def _tokenize_words(self, words):
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

    def _hidden_state(self, enc, layer):
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        return out.hidden_states[layer].squeeze(0)

    @staticmethod
    def _pool_words(hs, word_ids):
        buckets = defaultdict(list)
        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            buckets[wid].append(hs[i])
        keep = sorted(buckets.keys())
        if not keep:
            return torch.empty((0, hs.size(-1)), device=hs.device), keep
        reps = torch.stack([torch.stack(buckets[k]).mean(0) for k in keep])
        return reps, keep


class AlignmentDiagnostics:
    __slots__ = ("links", "kz_keep", "ru_keep", "p_ru_given_kz", "p_kz_given_ru")

    def __init__(self, links, kz_keep, ru_keep, p_ru_given_kz, p_kz_given_ru):
        self.links = links
        self.kz_keep = kz_keep
        self.ru_keep = ru_keep
        self.p_ru_given_kz = p_ru_given_kz
        self.p_kz_given_ru = p_kz_given_ru

    def iter_rows(
        self, kz_words, ru_words
    ):
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

    def to_dataframe(self, kz_words, ru_words):
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

    def link_probability(self, kz_idx, ru_idx):
        kz_lookup = {idx: i for i, idx in enumerate(self.kz_keep)}
        ru_lookup = {idx: j for j, idx in enumerate(self.ru_keep)}
        i = kz_lookup[kz_idx]
        j = ru_lookup[ru_idx]
        return float(min(self.p_ru_given_kz[i, j], self.p_kz_given_ru[i, j]))


def merge_kz_to_single_ru(
    kazakh_words,
    russian_words,
    links,
):
    alignment_index = defaultdict(list)
    for kaz_idx, ru_idx in links:
        alignment_index[ru_idx].append(kaz_idx)

    merged_spans = []
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

    result = []
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


NON_WORD_RE = re.compile(r"^(?:\W+|\d[\d\W]*)$")


def is_informative(token):
    token = token.strip().lower()
    if not token or NON_WORD_RE.fullmatch(token):
        return False
    return is_cyrillic_token(token)


def informative_link_share(
    details,
    kz_words,
    ru_words,
    *,
    gold_links=None,
):
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
    samples,
    aligner,
    *,
    layer=8,
    thresh=0.05,
    prob_threshold=0.2,
    kaz_key="kaz_sent",
    rus_key="rus_sent",
):
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
