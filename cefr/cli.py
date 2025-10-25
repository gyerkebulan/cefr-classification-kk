from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

from cefr import load_config
from cefr.pipeline import TextPipeline
from cefr.text_utils import tokenize_words
from cefr.data.silver import build_silver_labels
from cefr.training import (
    TextClassificationConfig,
    WordTransformerConfig,
    train_text_classifier,
    train_word_transformer,
)


def _dump(data: dict, pretty: bool) -> None:
    json.dump(data, sys.stdout, ensure_ascii=False, indent=2 if pretty else None)
    sys.stdout.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cefr.cli",
        description="CEFR utilities for Kazakh ↔ Russian pipelines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    align = subparsers.add_parser(
        "align",
        help="Align a Kazakh text with a Russian translation and report token links.",
    )
    align.add_argument("--kaz-text", required=True, help="Kazakh text to align.")
    align.add_argument(
        "--rus-text",
        help="Optional Russian translation. If omitted, the translator is used.",
    )
    align.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    word = subparsers.add_parser(
        "word",
        help="Predict CEFR level for a single Kazakh word.",
    )
    word.add_argument("--kaz-word", required=True, help="Kazakh word to score.")
    word.add_argument(
        "--rus-word",
        help="Optional Russian translation. If omitted, the translator is used.",
    )
    word.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    text = subparsers.add_parser(
        "text",
        help="Predict CEFR level for a Kazakh text passage.",
    )
    text.add_argument("--kaz-text", required=True, help="Kazakh text to score.")
    text.add_argument(
        "--rus-text",
        help="Optional Russian translation. If omitted, the translator is used.",
    )
    text.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    train_word = subparsers.add_parser(
        "train-word",
        help="Fine-tune a transformer to predict CEFR levels for Russian tokens.",
    )
    train_word.add_argument(
        "--dataset-path",
        default="data/labels/silver_word_labels.csv",
        help="CSV containing silver word labels (columns 'rus_item' and 'cefr').",
    )
    train_word.add_argument(
        "--output-dir",
        default="models/transformer_word_cefr",
        help="Directory where the fitted model and tokenizer will be stored.",
    )
    train_word.add_argument(
        "--text-column",
        default="rus_item",
        help="Name of the column containing the token text.",
    )
    train_word.add_argument(
        "--label-column",
        default="cefr",
        help="Name of the column containing CEFR labels.",
    )
    train_word.add_argument(
        "--parallel-path",
        default="data/parallel/kazparc_kz_ru.csv",
        help="Parallel corpus used to (re)generate silver labels when needed.",
    )
    train_word.add_argument(
        "--russian-cefr",
        default=None,
        help="Optional override for the Russian CEFR lexicon CSV.",
    )
    train_word.add_argument(
        "--rebuild",
        action="store_true",
        help="Regenerate silver labels before training, even if the dataset already exists.",
    )
    train_word.add_argument("--test-size", type=float, default=0.2, help="Validation split size.")
    train_word.add_argument("--random-state", type=int, default=42, help="Random seed for splits.")
    train_word.add_argument(
        "--model-name",
        default="cointegrated/rubert-tiny2",
        help="Hugging Face model name or path to fine-tune.",
    )
    train_word.add_argument("--max-length", type=int, default=16, help="Maximum tokenised length.")
    train_word.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    train_word.add_argument(
        "--train-batch-size",
        type=int,
        default=32,
        help="Per-device batch size for training.",
    )
    train_word.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
        help="Per-device batch size for evaluation.",
    )
    train_word.add_argument("--learning-rate", type=float, default=2e-5, help="Initial learning rate.")
    train_word.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay coefficient.")
    train_word.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio for scheduler.")
    train_word.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    silver = subparsers.add_parser(
        "silver",
        help="Generate silver word-level CEFR labels from a parallel corpus.",
    )
    silver.add_argument(
        "--size",
        default=10000
    )
    silver.add_argument(
        "--parallel-path",
        default="data/parallel/kazparc_kz_ru.csv",
        help="Parallel corpus CSV with columns 'kaz' and 'rus'.",
    )
    silver.add_argument(
        "--output-path",
        default="data/labels/silver_word_labels.csv",
        help="Where to write the generated silver labels CSV.",
    )
    silver.add_argument(
        "--russian-cefr",
        default=None,
        help="Optional path to the Russian CEFR lexicon CSV (defaults to config pipeline setting).",
    )

    train_text = subparsers.add_parser(
        "train-text",
        help="Train the bilingual CEFR text classifier.",
    )
    train_text.add_argument(
        "--dataset-path",
        default="data/text/en_ru_cefr_corpus.csv",
        help="CSV containing English/Russian texts with CEFR labels.",
    )
    train_text.add_argument(
        "--output-dir",
        default="models/text_classifier",
        help="Directory where the fitted model and metrics will be stored.",
    )
    train_text.add_argument("--test-size", type=float, default=0.2, help="Validation split size.")
    train_text.add_argument("--random-state", type=int, default=42, help="Random seed for splits.")
    train_text.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum number of TF-IDF features per language.",
    )
    train_text.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Maximum n-gram length for TF-IDF extraction.",
    )
    train_text.add_argument(
        "--include-english-text",
        action="store_true",
        help="Include English text features alongside Russian during training.",
    )
    train_text.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    return parser


def _run_align(pipeline: TextPipeline, args: argparse.Namespace) -> dict:
    prediction = pipeline.predict(args.kaz_text, russian_text=args.rus_text)

    kaz_tokens = tokenize_words(args.kaz_text)
    rus_tokens = tokenize_words(prediction.translation)
    diagnostics = pipeline.aligner.diagnostics(
        kaz_tokens,
        rus_tokens,
        layer=pipeline.config.alignment.layer,
        threshold=pipeline.config.alignment.threshold,
    )

    alignments: list[dict[str, object]] = []
    for row in diagnostics.iter_rows(kaz_tokens, rus_tokens):
        if not row.get("is_link"):
            continue
        alignments.append(
            {
                "kaz_index": row["kaz_index"],
                "kaz_token": row["kaz_token"],
                "rus_index": row["rus_index"],
                "rus_token": row["rus_token"],
                "probability": row["joint_prob"],
            }
        )

    return {
        "kaz_text": args.kaz_text,
        "rus_text": prediction.translation,
        "translation_used": args.rus_text is None,
        "alignments": alignments,
    }


def _run_word(pipeline: TextPipeline, args: argparse.Namespace) -> dict:
    prediction = pipeline.predict(args.kaz_word, russian_text=args.rus_word)
    info = prediction.level_for_word(args.kaz_word)
    if info is None:
        raise SystemExit("Could not determine CEFR level for the supplied word.")
    level, confidence, alignment = info
    return {
        "kaz_word": alignment.kazakh_token,
        "rus_token": alignment.russian_token,
        "translation_used": args.rus_word is None,
        "cefr_level": level,
        "confidence": confidence,
        "distribution": prediction.distribution,
    }


def _run_text(pipeline: TextPipeline, args: argparse.Namespace) -> dict:
    prediction = pipeline.predict(args.kaz_text, russian_text=args.rus_text)
    level = prediction.average_level
    confidence = float(prediction.distribution.get(level, 0.0))
    return {
        "kaz_text": args.kaz_text,
        "rus_text": prediction.translation,
        "translation_used": args.rus_text is None,
        "cefr_level": level,
        "confidence": confidence,
        "distribution": prediction.distribution,
    }


def _run_train_word(args: argparse.Namespace) -> dict:
    dataset_path = Path(args.dataset_path)
    if args.rebuild or not dataset_path.exists():
        parallel_path = Path(args.parallel_path)
        if not parallel_path.exists():
            raise FileNotFoundError(
                f"Parallel corpus not found at {parallel_path}. Provide --parallel-path or disable --rebuild."
            )
        cfg = load_config()
        pipeline = TextPipeline(config=cfg.pipeline)
        russian_cefr = Path(args.russian_cefr) if args.russian_cefr else Path(cfg.pipeline.russian_cefr_path)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        build_silver_labels(
            parallel_csv=parallel_path,
            rus_cefr=russian_cefr,
            out_csv=dataset_path,
            aligner=pipeline.aligner,
            alignment_config=cfg.pipeline.alignment,
        )
    if not dataset_path.exists():
        raise FileNotFoundError(f"Silver labels not found at {dataset_path}")

    config = WordTransformerConfig(
        dataset_path=dataset_path,
        output_dir=args.output_dir,
        text_column=args.text_column,
        label_column=args.label_column,
        test_size=args.test_size,
        random_state=args.random_state,
        model_name=args.model_name,
        max_length=args.max_length,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )
    return train_word_transformer(config)


def _run_train_text(args: argparse.Namespace) -> dict:
    include_english_text = getattr(args, "include_english_text", False)
    config = TextClassificationConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        include_english_text=include_english_text,
        include_russian_text=True,
    )
    result = train_text_classifier(config)
    return {
        "model_path": result["model_path"],
        "metrics_path": result["metrics_path"],
        "accuracy": result["accuracy"],
    }


def _run_generate_silver(args: argparse.Namespace) -> dict:
    cfg = load_config()
    pipeline = TextPipeline(config=cfg.pipeline)
    parallel_path = Path(args.parallel_path)
    if not parallel_path.exists():
        raise FileNotFoundError(f"Parallel corpus not found at {parallel_path}")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    russian_cefr = Path(args.russian_cefr) if args.russian_cefr else Path(cfg.pipeline.russian_cefr_path)
    build_silver_labels(
        parallel_csv=parallel_path,
        rus_cefr=russian_cefr,
        out_csv=output_path,
        aligner=pipeline.aligner,
        alignment_config=cfg.pipeline.alignment,
        size = args.size,
    )
    return {
        "output_path": str(output_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train-word":
        result = _run_train_word(args)
    elif args.command == "train-text":
        result = _run_train_text(args)
    elif args.command == "silver":
        result = _run_generate_silver(args)
    else:
        config = load_config()
        pipeline = TextPipeline(config=config.pipeline)
        if args.command == "align":
            result = _run_align(pipeline, args)
        elif args.command == "word":
            result = _run_word(pipeline, args)
        elif args.command == "text":
            result = _run_text(pipeline, args)
        else:  # pragma: no cover
            parser.error(f"Unknown command: {args.command}")
            return 2

    _dump(result, pretty=getattr(args, "pretty", False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
