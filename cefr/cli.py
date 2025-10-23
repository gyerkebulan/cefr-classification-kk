from __future__ import annotations

import argparse
import json
import sys

from cefr import load_config
from cefr.pipeline import TextPipeline
from cefr.text_utils import tokenize_words


def _dump(data: dict, pretty: bool) -> None:
    json.dump(
        data,
        sys.stdout,
        ensure_ascii=False,
        indent=2 if pretty else None,
    )
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
    align.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )

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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

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
