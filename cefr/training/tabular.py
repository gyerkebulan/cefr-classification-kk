import json
import argparse
from pathlib import Path
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


TEXT_COLUMNS = ("kaz", "rus")
NUMERIC_COLUMNS = (
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


class TabularTrainingConfig:
    __slots__ = ("train_path", "output_dir", "test_size", "random_state", "max_features", "ngram_max")

    def __init__(
        self,
        train_path,
        output_dir,
        test_size=0.2,
        random_state=42,
        max_features=3000,
        ngram_max=2,
    ):
        self.train_path = Path(train_path)
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.random_state = random_state
        self.max_features = max_features
        self.ngram_max = ngram_max


def _build_preprocessor(config):
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


def _build_model(config):
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


def _load_dataset(path):
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


def train_tabular_model(config):
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

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Train a CEFR classifier on the KazParC dataset with engineered features."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/text/kazparc_kz_ru_cefr_estimated.csv"),
        help="Path to the CSV file with Kazakh/Russian pairs and features.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/kazparc_tabular_cefr"),
        help="Where to save the fitted pipeline and metrics.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument(
        "--max-features",
        type=int,
        default=3000,
        help="Maximum number of TF-IDF features per language.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Maximum n-gram length for the TF-IDF extractor.",
    )
    parsed = parser.parse_args(args=args)
    return TabularTrainingConfig(
        train_path=parsed.train_path,
        output_dir=parsed.output_dir,
        test_size=parsed.test_size,
        random_state=parsed.random_state,
        max_features=parsed.max_features,
        ngram_max=parsed.ngram_max,
    )


def main(args=None):
    config = parse_args(args=args)
    result = train_tabular_model(config)
    print(f"Saved model to {result['model_path']}")
    print(f"Validation accuracy: {result['accuracy']:.3f}")
    print(f"Metrics report written to {result['metrics_path']}")


__all__ = ["TabularTrainingConfig", "train_tabular_model", "parse_args", "main"]


if __name__ == "__main__":
    main()
