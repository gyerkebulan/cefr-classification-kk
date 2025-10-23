from functools import lru_cache
from pathlib import Path

import pandas as pd

CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
DEFAULT_RUS_CEFR = Path("data/cefr/russian_cefr_sample.csv")


class RussianCefrRepository:
    """Lazy loader for the Russian lexicon → CEFR mapping."""

    __slots__ = ("path",)

    def __init__(self, path=DEFAULT_RUS_CEFR):
        self.path = Path(path)

    @property
    def mapping(self):
        return _load_russian_cefr_mapping(self.path)

    def lookup_level(self, token):
        token_norm = token.strip().lower()
        if not token_norm:
            return "Unknown"
        return self.mapping.get(token_norm, "Unknown")


@lru_cache(maxsize=4)
def _load_russian_cefr_mapping(path):
    if not path.exists():
        raise FileNotFoundError(f"Russian CEFR list not found at: {path}")
    csv = pd.read_csv(path)
    return {
        str(row.word).strip().lower(): str(row.level).strip()
        for _, row in csv.iterrows()
    }


__all__ = ["RussianCefrRepository", "DEFAULT_RUS_CEFR", "CEFR_LEVELS"]
