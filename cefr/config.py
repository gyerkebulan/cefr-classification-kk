from dataclasses import dataclass, field, is_dataclass
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path("config/default.yaml")


@dataclass(slots=True)
class TranslatorConfig:
    model_name: str = "issai/tilmash"
    device: str | int | None = None


@dataclass(slots=True)
class AlignmentConfig:
    model_name: str = "aneuraz/awesome-align-with-co"
    device: str | None = None
    layer: int = 8
    threshold: float = 0.05


@dataclass(slots=True)
class PipelineConfig:
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    russian_cefr_path: str = "data/cefr/russian_cefr_sample.csv"
    russian_model_dir: str | None = "models/ru_cefr_sentence"
    russian_weight: float = 0.6
    word_model_dir: str | None = "models/transformer_word_cefr"
    word_model_device: str | int | None = None


@dataclass(slots=True)
class NotebookConfig:
    """Encapsulates notebook defaults derived from the top-level pipeline config."""

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    use_ensemble: bool = False


def _load_yaml(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _merge_config(instance, values):
    for field_name, field_value in values.items():
        if not hasattr(instance, field_name):
            continue
        current = getattr(instance, field_name)
        if isinstance(field_value, dict) and is_dataclass(current):
            _merge_config(current, dict(field_value))
        else:
            setattr(instance, field_name, field_value)
    return instance


def load_config(path=None):
    """Load configuration from YAML into a NotebookConfig instance."""

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    data = _load_yaml(config_path)
    notebook_cfg = NotebookConfig()
    if data:
        _merge_config(notebook_cfg, dict(data))
    return notebook_cfg


__all__ = [
    "TranslatorConfig",
    "AlignmentConfig",
    "PipelineConfig",
    "NotebookConfig",
    "load_config",
    "DEFAULT_CONFIG_PATH",
]
