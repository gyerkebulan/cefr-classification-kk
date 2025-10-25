from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_DIR = Path("models/russian_word_cefr")
CEFR_LEVELS: Sequence[str] = ("A1", "A2", "B1", "B2", "C1", "C2")
_CACHE: Dict[Tuple[str, str], tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]] = {}

__all__ = [
    "CEFR_LEVELS",
    "predict_word_level",
    "predict_word_distribution",
    "predict_word_batch",
    "load_transformer_resources",
]


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def load_transformer_resources(
    model_dir: Path = DEFAULT_MODEL_DIR,
    *,
    device: str | torch.device | None = None,
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    """
    Load (and cache) the fine-tuned transformer and tokenizer for word-level CEFR prediction.
    """
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Transformer CEFR model not found at '{path}'. "
            "Train it first via `python -m cefr.cli train-word`."
        )

    resolved_device = _resolve_device(device)
    cache_key = (str(path.resolve()), str(resolved_device))
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(resolved_device)
    model.eval()
    _CACHE[cache_key] = (tokenizer, model, resolved_device)
    return tokenizer, model, resolved_device


def _prepare_words(words: Iterable[str]) -> List[str]:
    prepared: List[str] = []
    for word in words:
        cleaned = str(word).strip()
        if not cleaned:
            raise ValueError("Words for prediction must be non-empty strings.")
        prepared.append(cleaned)
    if not prepared:
        raise ValueError("At least one word must be provided for prediction.")
    return prepared


def predict_word_batch(
    words: Sequence[str],
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    device: str | torch.device | None = None,
    return_probabilities: bool = False,
) -> List[str] | List[Dict[str, float]]:
    """
    Predict CEFR levels (or probability distributions) for a batch of words.
    """
    tokenizer, model, resolved_device = load_transformer_resources(model_dir, device=device)
    prepared_words = _prepare_words(words)

    encoded = tokenizer(
        prepared_words,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(resolved_device)

    with torch.no_grad():
        logits = model(**encoded).logits

    id2label = model.config.id2label or {idx: label for idx, label in enumerate(CEFR_LEVELS)}

    if return_probabilities:
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return [
            {id2label.get(idx, CEFR_LEVELS[idx]): float(prob) for idx, prob in enumerate(row)}
            for row in probs
        ]

    preds = torch.argmax(logits, dim=-1).cpu().tolist()
    return [id2label.get(pred, CEFR_LEVELS[pred]) for pred in preds]


def predict_word_level(
    word: str,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    device: str | torch.device | None = None,
) -> str:
    """
    Predict the CEFR level for a single word.
    """
    return predict_word_batch(
        [word],
        model_dir=model_dir,
        device=device,
        return_probabilities=False,
    )[0]


def predict_word_distribution(
    word: str,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    device: str | torch.device | None = None,
) -> Dict[str, float]:
    """
    Return the CEFR probability distribution for a single word.
    """
    return predict_word_batch(
        [word],
        model_dir=model_dir,
        device=device,
        return_probabilities=True,
    )[0]
