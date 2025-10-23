import json
import argparse
from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cefr.text_features import compute_text_features

TEXT_EN_COLUMN = "text_en"
TEXT_RU_COLUMN = "text_ru"
LABEL_COLUMN = "label"


class TextClassificationConfig:
    __slots__ = (
        "dataset_path",
        "output_dir",
        "test_size",
        "random_state",
        "max_features",
        "ngram_max",
        "include_russian_text",
    )

    def __init__(
        self,
        dataset_path,
        output_dir,
        test_size=0.2,
        random_state=42,
        max_features=5000,
        ngram_max=2,
        include_russian_text=True,
    ):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.random_state = random_state
        self.max_features = max_features
        self.ngram_max = ngram_max
        self.include_russian_text = include_russian_text


def _ensure_required_columns(df):
    missing = [col for col in (TEXT_EN_COLUMN, LABEL_COLUMN) if col not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Dataset missing required columns: {joined}")
    if TEXT_RU_COLUMN not in df.columns:
        df[TEXT_RU_COLUMN] = ""


def _load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    _ensure_required_columns(df)
    df = df.dropna(subset=[LABEL_COLUMN])
    df[TEXT_EN_COLUMN] = df[TEXT_EN_COLUMN].fillna("")
    df[TEXT_RU_COLUMN] = df[TEXT_RU_COLUMN].fillna("")
    return df


def _compute_feature_frame(df):
    english_feats = _features_from_series(df[TEXT_EN_COLUMN], prefix="en")
    russian_feats = _features_from_series(df[TEXT_RU_COLUMN], prefix="ru")
    combined = pd.concat([english_feats, russian_feats], axis=1)
    combined["aligned_token_ratio"] = _safe_ratio(
        english_feats["en_token_count"], russian_feats["ru_token_count"]
    )
    combined["aligned_sentence_ratio"] = _safe_ratio(
        english_feats["en_sentence_count"], russian_feats["ru_sentence_count"]
    )
    combined["syllable_ratio"] = _safe_ratio(
        english_feats["en_syllable_count"], russian_feats["ru_syllable_count"]
    )
    return combined


def _features_from_series(series, *, prefix):
    features = series.astype(str).apply(lambda text: compute_text_features(text).as_dict())
    frame = pd.DataFrame(features.tolist())
    frame.columns = [f"{prefix}_{column}" for column in frame.columns]
    return frame


def _safe_ratio(numerator, denominator):
    denom = denominator.replace(0, pd.NA)
    ratio = numerator / denom
    return ratio.fillna(0.0)


def _build_preprocessor(
    config,
    text_columns,
    numeric_columns,
):
    text_features = [
        (
            f"tfidf_{column}",
            TfidfVectorizer(
                max_features=config.max_features,
                ngram_range=(1, config.ngram_max),
            ),
            column,
        )
        for column in text_columns
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
            ("numeric", numeric_pipeline, numeric_columns),
        ],
        sparse_threshold=0.3,
    )


def _build_model(
    config,
    text_columns,
    numeric_columns,
):
    preprocessor = _build_preprocessor(config, text_columns, numeric_columns)
    classifier = LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=2000,
        multi_class="multinomial",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )


def _prepare_training_frame(
    df,
    *,
    include_russian,
):
    features = _compute_feature_frame(df)
    df_reset = df.reset_index(drop=True)
    combined = pd.concat([df_reset, features], axis=1)
    text_columns = [TEXT_EN_COLUMN]
    if include_russian and TEXT_RU_COLUMN in combined.columns:
        text_columns.append(TEXT_RU_COLUMN)
    return combined, text_columns, list(features.columns)


def train_text_classifier(config):
    df = _load_dataset(config.dataset_path)
    training_frame, text_columns, numeric_columns = _prepare_training_frame(
        df,
        include_russian=config.include_russian_text,
    )

    X = training_frame.drop(columns=[LABEL_COLUMN])
    y = training_frame[LABEL_COLUMN].astype(str)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    pipeline = _build_model(config, text_columns, numeric_columns)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    accuracy = float(accuracy_score(y_val, y_pred))
    report = classification_report(
        y_val,
        y_pred,
        zero_division=0,
        output_dict=True,
    )
    labels = sorted(y.unique())
    conf_mtx = confusion_matrix(y_val, y_pred, labels=labels).tolist()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.output_dir / "text_classifier.joblib"
    metrics_path = config.output_dir / "text_classifier_metrics.json"

    dump(pipeline, model_path)
    metrics_path.write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "classification_report": report,
                "labels": labels,
                "confusion_matrix": conf_mtx,
                "class_distribution": y.value_counts(normalize=True).to_dict(),
                "config": {
                    "dataset_path": str(config.dataset_path),
                    "test_size": config.test_size,
                    "random_state": config.random_state,
                    "max_features": config.max_features,
                    "ngram_max": config.ngram_max,
                    "include_russian_text": config.include_russian_text,
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


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Train a CEFR classifier on bilingual English/Russian texts."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/text/en_ru_cefr_corpus.csv"),
        help="CSV file containing text_en, text_ru and label columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/en_ru_text_classifier"),
        help="Directory where the fitted model and metrics will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset to hold out for validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible train/validation splits.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum number of TF-IDF features per language.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Longest n-gram size to use for TF-IDF features.",
    )
    parser.add_argument(
        "--no-russian-text",
        action="store_true",
        help="Disable Russian text inputs (train on English only).",
    )
    parsed = parser.parse_args(args=args)
    return TextClassificationConfig(
        dataset_path=parsed.dataset_path,
        output_dir=parsed.output_dir,
        test_size=parsed.test_size,
        random_state=parsed.random_state,
        max_features=parsed.max_features,
        ngram_max=parsed.ngram_max,
        include_russian_text=not parsed.no_russian_text,
    )


def main(args=None):
    config = parse_args(args=args)
    result = train_text_classifier(config)
    print(f"Saved model to {result['model_path']}")
    print(f"Validation accuracy: {result['accuracy']:.3f}")
    print(f"Metrics report written to {result['metrics_path']}")


__all__ = [
    "TextClassificationConfig",
    "train_text_classifier",
    "parse_args",
    "main",
]


if __name__ == "__main__":
    main()
