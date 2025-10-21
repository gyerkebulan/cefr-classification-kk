from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from cefr.training import TabularTrainingConfig, train_tabular_model


def parse_args(args: Sequence[str] | None = None) -> TabularTrainingConfig:
    parser = argparse.ArgumentParser(description="Train a CEFR classifier on the KazParC dataset.")
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


def main(args: Sequence[str] | None = None) -> None:
    config = parse_args(args)
    result = train_tabular_model(config)
    print(f"Saved model to {result['model_path']}")
    print(f"Validation accuracy: {result['accuracy']:.3f}")
    print(f"Metrics report written to {result['metrics_path']}")


if __name__ == "__main__":
    main()
