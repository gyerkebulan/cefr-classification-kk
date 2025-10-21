from __future__ import annotations

from typing import Dict, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TranslationPipeline

from .config import TranslatorConfig


TranslatorCacheKey = Tuple[str, int]
_TRANSLATORS: Dict[TranslatorCacheKey, "Translator"] = {}


def _resolve_pipeline_device(device: int | str | None) -> int:
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
    """Thin wrapper around the HuggingFace translation pipeline."""

    def __init__(self, config: TranslatorConfig | None = None) -> None:
        if config is None:
            config = TranslatorConfig()
        self.model_name = config.model_name
        self.device = _resolve_pipeline_device(config.device)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
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


def get_translator(config: TranslatorConfig | None = None) -> Translator:
    cfg = config or TranslatorConfig()
    resolved_device = _resolve_pipeline_device(cfg.device)
    cache_key = (cfg.model_name, resolved_device)
    translator = _TRANSLATORS.get(cache_key)
    if translator is None:
        translator = Translator(cfg)
        _TRANSLATORS[cache_key] = translator
    return translator


__all__ = ["Translator", "get_translator"]
