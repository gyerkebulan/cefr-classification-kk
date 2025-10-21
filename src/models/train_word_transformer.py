from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.utils import cefr_label_to_id, set_seed

DEFAULT_CSV = Path("data/labels/silver_word_labels.csv")
DEFAULT_MODEL_NAME = "cointegrated/rubert-tiny2"
DEFAULT_OUT_DIR = Path("models/transformer_word_cefr")
CEFR_LEVELS: Sequence[str] = ("A1", "A2", "B1", "B2", "C1", "C2")


def _load_dataframe(csv_path: Path, text_column: str = "rus_item") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found in {csv_path}")
    df = df[df["cefr"].isin(CEFR_LEVELS)]
    if df.empty:
        raise RuntimeError(
            "No usable rows found after filtering for CEFR levels. "
            "Ensure the dataset contains labelled entries."
        )
    df = df.copy()
    df["label"] = df["cefr"].apply(cefr_label_to_id)
    df["text"] = df[text_column].astype(str).str.strip()
    df = df[df["text"] != ""]
    if df.empty:
        raise RuntimeError("All candidate texts are empty after preprocessing.")
    return df[["text", "label"]]


def load_dataset_splits(
    csv_path: Path,
    *,
    text_column: str = "rus_item",
    test_size: float = 0.2,
    seed: int = 42,
) -> DatasetDict:
    df = _load_dataframe(csv_path, text_column=text_column)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    def to_dataset(frame: pd.DataFrame) -> Dataset:
        dataset = Dataset.from_pandas(frame.reset_index(drop=True))
        drop_cols = [
            col for col in dataset.column_names if col not in {"text", "label"}
        ]
        if drop_cols:
            dataset = dataset.remove_columns(drop_cols)
        return dataset

    return DatasetDict(
        train=to_dataset(train_df),
        test=to_dataset(test_df),
    )


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    *,
    max_length: int = 16,
) -> DatasetDict:
    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list]:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        enc["labels"] = batch["label"]
        return enc

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=["text", "label"],
        desc="Tokenising dataset",
    )
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return tokenized


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}


def train_transformer_word_classifier(
    csv_path: Path = DEFAULT_CSV,
    *,
    text_column: str = "rus_item",
    model_name: str = DEFAULT_MODEL_NAME,
    out_dir: Path = DEFAULT_OUT_DIR,
    epochs: int = 5,
    train_batch_size: int = 32,
    eval_batch_size: int = 64,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    max_length: int = 16,
    seed: int = 42,
) -> Path:
    set_seed(seed)

    dataset = load_dataset_splits(csv_path, text_column=text_column, test_size=0.2, seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length=max_length)

    label2id = {level: idx for idx, level in enumerate(CEFR_LEVELS)}
    id2label = {idx: level for level, idx in label2id.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CEFR_LEVELS),
        label2id=label2id,
        id2label=id2label,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=20,
        logging_dir=str(out_dir / "logs"),
        report_to=[],
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    print(f"Saved transformer model and tokenizer to {out_dir}")
    return out_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a transformer on Russian word CEFR labels."
    )
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to the silver labels CSV.")
    parser.add_argument(
        "--text-column",
        default="rus_item",
        help="Column containing the word/token to classify.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name or local path to fine-tune.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory to store the fine-tuned model artefacts.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=32,
        help="Per-device batch size for training.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
        help="Per-device batch size for evaluation.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-5, help="Initial learning rate."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.06,
        help="Warmup ratio for the learning rate scheduler.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=16,
        help="Maximum tokenised length for inputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting and training.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    train_transformer_word_classifier(
        csv_path=Path(args.csv),
        text_column=args.text_column,
        model_name=args.model_name,
        out_dir=Path(args.out_dir),
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
