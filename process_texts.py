from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc]

from cefr import load_config
from cefr.models.word_transformer import DEFAULT_MODEL_DIR, predict_word_batch
from cefr.text_utils import tokenize_words

CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    return frame


def _average_distribution(distributions: Iterable[dict[str, float]]) -> dict[str, float]:
    agg = {level: 0.0 for level in CEFR_LEVELS}
    count = 0
    for dist in distributions:
        count += 1
        for level in CEFR_LEVELS:
            agg[level] += float(dist.get(level, 0.0))

    if count == 0:
        return agg

    for level in CEFR_LEVELS:
        agg[level] /= count
    return agg


def _predict_paragraph(tokens: list[str], model_dir: Path, device: str | None) -> tuple[dict[str, float], str, float]:
    if not tokens:
        empty = {level: 0.0 for level in CEFR_LEVELS}
        return empty, "Unknown", 0.0

    distributions = predict_word_batch(
        tokens,
        model_dir=model_dir,
        device=device,
        return_probabilities=True,
    )
    averaged = _average_distribution(distributions)
    label = max(averaged, key=averaged.get, default="Unknown")
    confidence = float(averaged.get(label, 0.0))
    return averaged, label, confidence


def _iter_with_progress(iterable, *, description: str) -> Iterable:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=description)


def process_paragraphs(input_path: Path, output_path: Path, model_dir: Path, device: str | None) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = _normalise_columns(df)

    required_columns = {"text_id", "paragraph", "p_ru"}
    if missing := required_columns - set(df.columns):
        raise ValueError(f"Input CSV is missing required columns: {', '.join(sorted(missing))}")

    records: list[dict[str, object]] = []

    for row in _iter_with_progress(df.itertuples(index=False), description="Scoring paragraphs"):
        kz_text = str(getattr(row, "paragraph", "")).strip()
        ru_text = str(getattr(row, "p_ru", "")).strip()
        text_id = getattr(row, "text_id")

        tokens = tokenize_words(ru_text)
        averaged, label, confidence = _predict_paragraph(tokens, model_dir, device)

        record: dict[str, object] = {
            "text_id": text_id,
            "paragraph": kz_text,
            "p_ru": ru_text,
            "token_count": len(tokens),
            "predicted_level": label,
            "predicted_confidence": confidence,
        }
        for level in CEFR_LEVELS:
            record[f"prob_{level}"] = averaged[level]
        records.append(record)

    paragraph_df = pd.DataFrame(records)
    paragraph_df.to_csv(output_path, index=False, encoding="utf-8")
    return paragraph_df


def aggregate_texts(paragraph_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    aggregates: list[dict[str, object]] = []

    grouped = paragraph_df.groupby("text_id", sort=False)
    iterator = _iter_with_progress(grouped, description="Aggregating texts")
    for text_id, group in iterator:
        token_total = group["token_count"].sum()
        if token_total <= 0:
            token_total = 1

        summed = {level: 0.0 for level in CEFR_LEVELS}
        for _, row in group.iterrows():
            weight = row["token_count"] or 1
            for level in CEFR_LEVELS:
                summed[level] += row[f"prob_{level}"] * weight

        for level in CEFR_LEVELS:
            summed[level] /= token_total

        final_level = max(summed, key=summed.get, default="Unknown")

        aggregates.append(
            {
                "text_id": text_id,
                "kz_text_joined": " ".join(group["paragraph"].astype(str)),
                "ru_text_joined": " ".join(group["p_ru"].astype(str)),
                "final_cefr_level": final_level,
            }
        )

    aggregated_df = pd.DataFrame(aggregates)
    aggregated_df.to_csv(output_path, index=False, encoding="utf-8")
    return aggregated_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process texts into paragraph and document-level CEFR predictions.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/bilim_articles_300_final.csv"),
        help="Path to the CSV containing paragraphs.",
    )
    parser.add_argument(
        "--paragraph-output",
        type=Path,
        default=Path("paragraphs_with_preds.csv"),
        help="Where to write per-paragraph predictions.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=Path("texts_aggregated.csv"),
        help="Where to write aggregated text predictions.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optional override for the word-level CEFR model directory.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device hint passed to the transformer (e.g. 'cpu', 'cuda', 'cuda:0').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()

    model_dir = args.model_dir or cfg.pipeline.word_model_dir or DEFAULT_WORD_MODEL_DIR
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Word-level CEFR model not found at '{model_dir}'. "
            "Train it via `python -m cefr.cli train-word` before running this script."
        )

    paragraph_df = process_paragraphs(args.input, args.paragraph_output, model_dir, args.device)
    aggregate_texts(paragraph_df, args.text_output)


if __name__ == "__main__":
    main()
