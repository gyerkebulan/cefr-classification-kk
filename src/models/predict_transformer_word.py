from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import cefr_id_to_label

DEFAULT_MODEL_DIR = Path("models/transformer_word_cefr")
CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
_CACHE: Dict[Tuple[str, str], tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]] = {}
__all__ = [
    "CEFR_LEVELS",
    "predict_transformer_word",
    "predict_transformer_distribution",
    "predict_transformer_batch",
    "load_transformer_resources",
]


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _load_resources(
    model_dir: Path = DEFAULT_MODEL_DIR,
    *,
    device: str | torch.device | None = None,
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Transformer CEFR model not found at '{path}'. "
            "Train it first via `python -m src.models.train_word_transformer`."
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


def predict_transformer_batch(
    words: Sequence[str],
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    device: str | torch.device | None = None,
    return_probabilities: bool = False,
) -> List[str] | List[Dict[str, float]]:
    """
    Predict CEFR levels for a batch of words.

    Parameters
    ----------
    words:
        Sequence of individual tokens/words to classify.
    model_dir:
        Directory containing the fine-tuned transformer weights and tokenizer.
    device:
        Optional device override (e.g., ``"cuda"`` or ``torch.device("cpu")``).
    return_probabilities:
        When ``True`` return per-class probability distributions instead of labels.
    """

    tokenizer, model, resolved_device = _load_resources(model_dir, device=device)
    prepared_words = _prepare_words(words)

    encoded = tokenizer(
        prepared_words,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(resolved_device)

    with torch.no_grad():
        logits = model(**encoded).logits

    if return_probabilities:
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return [
            {level: float(prob) for level, prob in zip(CEFR_LEVELS, row)}
            for row in probs
        ]

    preds = torch.argmax(logits, dim=-1).cpu().tolist()
    return [cefr_id_to_label(pred) for pred in preds]


def predict_transformer_word(
    word: str,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    device: str | torch.device | None = None,
) -> str:
    return predict_transformer_batch(
        [word],
        model_dir=model_dir,
        device=device,
        return_probabilities=False,
    )[0]


def load_transformer_resources(
    model_dir: Path = DEFAULT_MODEL_DIR,
    *,
    device: str | torch.device | None = None,
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    """
    Public wrapper around the cached loader so notebooks can reuse the model
    without reloading weights on every call.
    """
    return _load_resources(model_dir=model_dir, device=device)


def predict_transformer_distribution(
    word: str,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    device: str | torch.device | None = None,
) -> Dict[str, float]:
    """
    Return the full CEFR probability distribution for a single word.
    """
    return predict_transformer_batch(
        [word],
        model_dir=model_dir,
        device=device,
        return_probabilities=True,
    )[0]
