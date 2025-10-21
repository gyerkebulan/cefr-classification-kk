from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import pandas as pd


CEFR_LEVELS: tuple[str, ...] = ("A1", "A2", "B1", "B2", "C1", "C2")
DEFAULT_RUS_CEFR = Path("data/cefr/russian_cefr_sample.csv")


@dataclass(slots=True)
class RussianCefrRepository:
    """Lazy loader for the Russian lexicon → CEFR mapping."""

    path: Path = DEFAULT_RUS_CEFR

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    @property
    def mapping(self) -> Mapping[str, str]:
        return _load_russian_cefr_mapping(self.path)

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


__all__ = ["RussianCefrRepository", "DEFAULT_RUS_CEFR", "CEFR_LEVELS"]
