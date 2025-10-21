from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

import pandas as pd

DEFAULT_RUS_CEFR = Path("data/cefr/russian_cefr_sample.csv")


class RussianCefrRepository:
    """Lazy loader and accessor for the Russian word → CEFR mapping."""

    def __init__(self, path: str | Path = DEFAULT_RUS_CEFR) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def mapping(self) -> Mapping[str, str]:
        return _load_russian_cefr_mapping(self._path)

    def lookup_level(self, token: str) -> str:
        token_norm = token.strip().lower()
        if not token_norm:
            return "Unknown"
        return self.mapping.get(token_norm, "Unknown")


@lru_cache(maxsize=4)
def _load_russian_cefr_mapping(path: Path) -> Mapping[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Russian CEFR list not found at: {path}")
    csv = pd.read_csv(path)
    return {
        str(row.word).strip().lower(): str(row.level).strip()
        for _, row in csv.iterrows()
    }


__all__ = ["DEFAULT_RUS_CEFR", "RussianCefrRepository"]
