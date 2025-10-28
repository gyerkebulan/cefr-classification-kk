import json
import argparse
from collections import Counter
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
from cefr.text_utils import tokenize_words
from cefr.models import WORD_CEFR_LEVELS, predict_word_batch

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
        "include_english_text",
        "include_russian_text",
        "include_word_distribution",
        "word_model_dir",
    )

    def __init__(
        self,
        dataset_path,
        output_dir,
        test_size=0.2,
        random_state=42,
        max_features=5000,
        ngram_max=2,
        include_english_text=True,
        include_russian_text=True,
        include_word_distribution=True,
        word_model_dir=None,
    ):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.random_state = random_state
        self.max_features = max_features
        self.ngram_max = ngram_max
        self.include_english_text = include_english_text
        self.include_russian_text = include_russian_text
        self.include_word_distribution = include_word_distribution
        self.word_model_dir = Path(word_model_dir) if word_model_dir is not None else None


def _ensure_required_columns(df, *, require_english, require_russian):
    required = [LABEL_COLUMN]
    if require_english:
        required.append(TEXT_EN_COLUMN)
    if require_russian:
        required.append(TEXT_RU_COLUMN)
    missing = [col for col in required if col not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Dataset missing required columns: {joined}")
    if TEXT_EN_COLUMN not in df.columns:
        df[TEXT_EN_COLUMN] = ""
    if TEXT_RU_COLUMN not in df.columns:
        df[TEXT_RU_COLUMN] = ""


def _load_dataset(path, *, require_english, require_russian):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    _ensure_required_columns(
        df,
        require_english=require_english,
        require_russian=require_russian,
    )
    df = df.dropna(subset=[LABEL_COLUMN])
    df[TEXT_EN_COLUMN] = df[TEXT_EN_COLUMN].fillna("")
    df[TEXT_RU_COLUMN] = df[TEXT_RU_COLUMN].fillna("")
    return df


def _compute_feature_frame(
    df,
    *,
    include_english,
    include_russian,
    include_word_distribution,
    word_model_dir,
):
    frames = []
    english_feats = None
    russian_feats = None
    word_cefr_feats = None

    if include_english:
        english_feats = _features_from_series(df[TEXT_EN_COLUMN], prefix="en")
        frames.append(english_feats)
    if include_russian:
        russian_feats = _features_from_series(df[TEXT_RU_COLUMN], prefix="ru")
        frames.append(russian_feats)
    if include_word_distribution:
        if word_model_dir is None:
            raise ValueError(
                "word_model_dir must be provided when include_word_distribution is True."
            )
        target_series = df[TEXT_RU_COLUMN] if TEXT_RU_COLUMN in df.columns else pd.Series(
            ["" for _ in range(len(df))],
            index=df.index,
        )
        word_cefr_feats = _word_distribution_features(target_series, word_model_dir)
        frames.append(word_cefr_feats)

    if frames:
        combined = pd.concat(frames, axis=1)
    else:
        combined = pd.DataFrame(index=df.index)

    if english_feats is not None and russian_feats is not None:
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


def _empty_word_feature_row():
    feature_row = {
        **{f"word_cefr_prob_{level}": 0.0 for level in WORD_CEFR_LEVELS},
        **{f"word_cefr_ratio_{level}": 0.0 for level in WORD_CEFR_LEVELS},
        "word_cefr_token_count": 0.0,
    }
    return feature_row


def _word_distribution_features(series, model_dir):
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Word-level CEFR model not found at {model_path}. "
            "Train it via `python -m cefr.cli train-word` or disable word distribution features."
        )

    rows = []
    for text in series.astype(str):
        tokens = tokenize_words(text)
        if not tokens:
            rows.append(_empty_word_feature_row())
            continue
        try:
            distributions = predict_word_batch(
                tokens,
                model_dir=model_path,
                return_probabilities=True,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Failed to load word-level CEFR model from {model_path}. "
                "Ensure the model directory exists or disable word distribution features."
            ) from exc

        total = len(distributions)
        if total == 0:
            rows.append(_empty_word_feature_row())
            continue

        sum_probs = {level: 0.0 for level in WORD_CEFR_LEVELS}
        top_counts = Counter()
        for dist in distributions:
            if not dist:
                continue
            top_level = max(dist, key=dist.get)
            top_counts[top_level] += 1
            for level in WORD_CEFR_LEVELS:
                sum_probs[level] += float(dist.get(level, 0.0))

        feature_row = {}
        for level in WORD_CEFR_LEVELS:
            feature_row[f"word_cefr_prob_{level}"] = (
                sum_probs[level] / total if total else 0.0
            )
            feature_row[f"word_cefr_ratio_{level}"] = (
                top_counts.get(level, 0) / total if total else 0.0
            )
        feature_row["word_cefr_token_count"] = float(total)
        rows.append(feature_row)

    frame = pd.DataFrame(rows, index=series.index)
    # Ensure consistent column order even if dataframe was empty
    for level in WORD_CEFR_LEVELS:
        prob_col = f"word_cefr_prob_{level}"
        ratio_col = f"word_cefr_ratio_{level}"
        if prob_col not in frame.columns:
            frame[prob_col] = 0.0
        if ratio_col not in frame.columns:
            frame[ratio_col] = 0.0
    if "word_cefr_token_count" not in frame.columns:
        frame["word_cefr_token_count"] = 0.0
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
    include_english,
    include_russian,
    include_word_distribution,
    word_model_dir,
):
    features = _compute_feature_frame(
        df,
        include_english=include_english,
        include_russian=include_russian,
        include_word_distribution=include_word_distribution,
        word_model_dir=word_model_dir,
    )
    df_reset = df.reset_index(drop=True)
    combined = pd.concat([df_reset, features], axis=1)
    text_columns = []
    if include_english:
        text_columns.append(TEXT_EN_COLUMN)
    if include_russian and TEXT_RU_COLUMN in combined.columns:
        text_columns.append(TEXT_RU_COLUMN)
    if not text_columns:
        raise ValueError("At least one language column must be included for training.")
    return combined, text_columns, list(features.columns)


def train_text_classifier(config):
    if config.include_word_distribution and config.word_model_dir is None:
        raise ValueError(
            "word_model_dir must be set when include_word_distribution is True. "
            "Provide a trained word-level CEFR model or disable the feature."
        )
    df = _load_dataset(
        config.dataset_path,
        require_english=config.include_english_text,
        require_russian=config.include_russian_text,
    )
    training_frame, text_columns, numeric_columns = _prepare_training_frame(
        df,
        include_english=config.include_english_text,
        include_russian=config.include_russian_text,
        include_word_distribution=config.include_word_distribution,
        word_model_dir=config.word_model_dir,
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
                    "include_english_text": config.include_english_text,
                    "include_russian_text": config.include_russian_text,
                    "include_word_distribution": config.include_word_distribution,
                    "word_model_dir": str(config.word_model_dir) if config.word_model_dir else None,
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
        "--no-english-text",
        action="store_true",
        help="Disable English text inputs (train on Russian only).",
    )
    parser.add_argument(
        "--no-russian-text",
        action="store_true",
        help="Disable Russian text inputs (train on English only).",
    )
    parser.add_argument(
        "--word-model-dir",
        type=Path,
        default=Path("models/transformer_word_cefr"),
        help="Directory containing the word-level CEFR model used for token distributions.",
    )
    parser.add_argument(
        "--no-word-distribution",
        action="store_true",
        help="Disable word-level CEFR distribution features.",
    )
    parsed = parser.parse_args(args=args)
    return TextClassificationConfig(
        dataset_path=parsed.dataset_path,
        output_dir=parsed.output_dir,
        test_size=parsed.test_size,
        random_state=parsed.random_state,
        max_features=parsed.max_features,
        ngram_max=parsed.ngram_max,
        include_english_text=not parsed.no_english_text,
        include_russian_text=not parsed.no_russian_text,
        include_word_distribution=not parsed.no_word_distribution,
        word_model_dir=parsed.word_model_dir,
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
