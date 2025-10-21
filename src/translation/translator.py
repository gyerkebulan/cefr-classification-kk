from __future__ import annotations

from typing import Dict, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TranslationPipeline

MODEL_NAME = "issai/tilmash"


def _resolve_pipeline_device(device: int | str | None) -> int:
    """Normalise device hints to the indices expected by `TranslationPipeline`."""

    if device is None:
        return 0 if torch.cuda.is_available() else -1
    if isinstance(device, int):
        return device
    device_lower = device.lower()
    if device_lower == "cpu":
        return -1
    if device_lower == "cuda":
        return 0
    if device_lower.startswith("cuda:"):
        return int(device_lower.split(":", 1)[1])
    return int(device)


class Translator:
    def __init__(self, device: int | str | None = None, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.device = _resolve_pipeline_device(device)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # NLLB/Tilmash supports these codes per model card:
        # kaz_Cyrl (Kazakh), rus_Cyrl (Russian)
        self.pipeline = TranslationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            src_lang="kaz_Cyrl",
            tgt_lang="rus_Cyrl",
            max_length=1000,
            device=self.device,
        )

    def translate(self, text: str) -> str:
        return self.pipeline(text)[0]["translation_text"]


_TRANSLATOR_CACHE: Dict[Tuple[str, int], Translator] = {}


def get_translator(device: int | str | None = None, model_name: str = MODEL_NAME) -> Translator:
    """Return a cached `Translator` instance so the model loads only once."""

    resolved_device = _resolve_pipeline_device(device)
    cache_key = (model_name, resolved_device)
    if cache_key not in _TRANSLATOR_CACHE:
        _TRANSLATOR_CACHE[cache_key] = Translator(device=resolved_device, model_name=model_name)
    return _TRANSLATOR_CACHE[cache_key]


__all__ = ["Translator", "get_translator"]
