from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path("config/default.yaml")


class TranslatorConfig:
    __slots__ = ("model_name", "device")

    def __init__(self, model_name="issai/tilmash", device=None):
        self.model_name = model_name
        self.device = device


class AlignmentConfig:
    __slots__ = ("model_name", "device", "layer", "threshold")

    def __init__(
        self,
        model_name="aneuraz/awesome-align-with-co",
        device=None,
        layer=8,
        threshold=0.05,
    ):
        self.model_name = model_name
        self.device = device
        self.layer = layer
        self.threshold = threshold


class PipelineConfig:
    __slots__ = ("translator", "alignment", "russian_cefr_path", "russian_model_dir", "russian_weight")

    def __init__(
        self,
        translator=None,
        alignment=None,
        russian_cefr_path="data/cefr/russian_cefr_sample.csv",
        russian_model_dir="models/ru_cefr_sentence",
        russian_weight=0.6,
    ):
        self.translator = translator or TranslatorConfig()
        self.alignment = alignment or AlignmentConfig()
        self.russian_cefr_path = russian_cefr_path
        self.russian_model_dir = russian_model_dir
        self.russian_weight = russian_weight


class NotebookConfig:
    """Encapsulates notebook defaults derived from the top-level pipeline config."""

    __slots__ = ("pipeline", "use_ensemble")

    def __init__(self, pipeline=None, use_ensemble=False):
        self.pipeline = pipeline or PipelineConfig()
        self.use_ensemble = use_ensemble


def _load_yaml(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _is_config_like(obj):
    return hasattr(obj, "__dict__") or hasattr(obj, "__slots__")


def _merge_config(instance, values):
    for field_name, field_value in values.items():
        if not hasattr(instance, field_name):
            continue
        current = getattr(instance, field_name)
        if isinstance(field_value, dict) and _is_config_like(current):
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
