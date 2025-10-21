from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src.data.repositories import DEFAULT_RUS_CEFR, RussianCefrRepository


def load_russian_cefr_mapping(path: str | Path = DEFAULT_RUS_CEFR) -> Mapping[str, str]:
    """Return a lemma → CEFR level mapping loaded from CSV."""

    return RussianCefrRepository(path).mapping


__all__ = ["DEFAULT_RUS_CEFR", "load_russian_cefr_mapping"]
