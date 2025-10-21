"""Core domain objects and services used across the CEFR pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .entities import PhraseAlignment, TextCefrPrediction

__all__ = [
    "PhraseAlignment",
    "TextCefrPrediction",
    "AlignmentService",
    "TranslationService",
    "CefrScorer",
    "TextCefrPipeline",
]


if TYPE_CHECKING:  # pragma: no cover - for static typing only
    from .services import AlignmentService, CefrScorer, TextCefrPipeline, TranslationService


def __getattr__(name: str) -> Any:
    if name in {"AlignmentService", "TranslationService", "CefrScorer", "TextCefrPipeline"}:
        from .services import AlignmentService, CefrScorer, TextCefrPipeline, TranslationService

        exports = {
            "AlignmentService": AlignmentService,
            "TranslationService": TranslationService,
            "CefrScorer": CefrScorer,
            "TextCefrPipeline": TextCefrPipeline,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
