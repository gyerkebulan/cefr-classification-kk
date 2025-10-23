"""
Top-level package exposing the main CEFR pipeline primitives.

This module provides a compact import surface so callers do not need to know the
exact internal layout. The public API is intentionally small; favour these
entry points instead of importing deep modules.
"""

from .config import AlignmentConfig, PipelineConfig, TranslatorConfig, load_config
from .pipeline import EnsemblePrediction, EnsemblePipeline, TextPipeline

__all__ = [
    "AlignmentConfig",
    "PipelineConfig",
    "TranslatorConfig",
    "load_config",
    "TextPipeline",
    "EnsemblePipeline",
    "EnsemblePrediction",
]
