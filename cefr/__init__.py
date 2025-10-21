"""
Top-level package exposing the main CEFR pipeline primitives.

This module provides a compact import surface so callers do not need to know the
exact internal layout. The public API is intentionally small; favour these
entry points instead of importing deep modules.
"""

from .config import (
    AlignmentConfig,
    NotebookConfig,
    PipelineConfig,
    TranslatorConfig,
    load_config,
)
from .pipeline import (
    EnsemblePrediction,
    EnsemblePipeline,
    TextPipeline,
)
from .notebook import (
    NotebookPrediction,
    predict_notebook_view,
    rows_to_dataframe,
)

__all__ = [
    "AlignmentConfig",
    "NotebookConfig",
    "PipelineConfig",
    "TranslatorConfig",
    "load_config",
    "TextPipeline",
    "EnsemblePipeline",
    "EnsemblePrediction",
    "NotebookPrediction",
    "predict_notebook_view",
    "rows_to_dataframe",
]
