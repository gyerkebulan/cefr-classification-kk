from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TEXT_COLUMNS: tuple[str, str] = ("kaz", "rus")
NUMERIC_COLUMNS: tuple[str, ...] = (
    "difficulty_score",
    "avg_sent_len_mean",
    "avg_word_len_mean",
    "ttr_mean",
    "long_ratio_mean",
    "align_diff",
    "words_mean",
)
TARGET_COLUMN = "predicted_cefr"
SECONDARY_TARGET_COLUMN = "predicted_cefr_int"


@dataclass(slots=True)
class TabularTrainingConfig:
    train_path: Path
    output_dir: Path
    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 3000
    ngram_max: int = 2

    def __post_init__(self) -> None:
        self.train_path = Path(self.train_path)
        self.output_dir = Path(self.output_dir)


def _build_preprocessor(config: TabularTrainingConfig) -> ColumnTransformer:
    text_features = [
        (
            f"tfidf_{column}",
            TfidfVectorizer(
                max_features=config.max_features,
                ngram_range=(1, config.ngram_max),
            ),
            column,
        )
        for column in TEXT_COLUMNS
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            *text_features,
            ("numeric", numeric_pipeline, list(NUMERIC_COLUMNS)),
        ],
        sparse_threshold=0.3,
    )


def _build_model(config: TabularTrainingConfig) -> Pipeline:
    preprocessor = _build_preprocessor(config)
    classifier = LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=2000,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )


def _load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in (*TEXT_COLUMNS, TARGET_COLUMN) if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset {path} is missing required columns: {', '.join(missing)}")
    df = df.dropna(subset=[TARGET_COLUMN])
    for column in TEXT_COLUMNS:
        df[column] = df[column].fillna("")
    if SECONDARY_TARGET_COLUMN in df.columns:
        df = df.drop(columns=[SECONDARY_TARGET_COLUMN])
    return df


def train_tabular_model(config: TabularTrainingConfig) -> dict[str, object]:
    df = _load_dataset(config.train_path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(str)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    pipeline = _build_model(config)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    accuracy = float(accuracy_score(y_val, y_pred))
    report = classification_report(
        y_val,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.output_dir / "model.joblib"
    metrics_path = config.output_dir / "metrics.json"

    dump(pipeline, model_path)
    metrics_path.write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "classification_report": report,
                "labels": sorted(set(y)),
                "config": {
                    "train_path": str(config.train_path),
                    "test_size": config.test_size,
                    "random_state": config.random_state,
                    "max_features": config.max_features,
                    "ngram_max": config.ngram_max,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "accuracy": accuracy,
    }


__all__ = ["TabularTrainingConfig", "train_tabular_model"]
