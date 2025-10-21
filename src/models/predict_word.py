from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import cefr_id_to_label

MODEL_DIR = "models/word_cefr"

_RESOURCE_CACHE: Dict[Tuple[str, str], tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]] = {}


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _load_resources(
    model_dir: str | Path = MODEL_DIR,
    *,
    device: str | torch.device | None = None,
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    path = Path(model_dir)
    resolved_device = _resolve_device(device)
    cache_key = (str(path.resolve()), str(resolved_device))
    if cache_key in _RESOURCE_CACHE:
        return _RESOURCE_CACHE[cache_key]

    if not path.exists():
        raise FileNotFoundError(
            f"Word CEFR classifier weights not found at '{path}'. "
            "Run `python -m src.models.train_word` to train the model or "
            "provide a custom directory via `predict_word(..., model_dir=...)`."
        )

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(resolved_device)
    model.eval()
    _RESOURCE_CACHE[cache_key] = (tokenizer, model, resolved_device)
    return tokenizer, model, resolved_device


def predict_word(
    word: str,
    *,
    model_dir: str | Path = MODEL_DIR,
    device: str | torch.device | None = None,
) -> str:
    tokenizer, model, resolved_device = _load_resources(model_dir, device=device)
    enc = tokenizer(word, return_tensors="pt", truncation=True).to(resolved_device)
    with torch.no_grad():
        logits = model(**enc).logits
        pred = int(torch.argmax(logits, -1).item())
    return cefr_id_to_label(pred)
