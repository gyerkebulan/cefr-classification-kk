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

DEFAULT_CSV = "data/labels/silver_word_labels.csv"
DEFAULT_MODEL_NAME = "kz-transformers/kaz-roberta-conversational"
DEFAULT_OUT_DIR = "models/word_cefr"
CEFR_LEVELS: Sequence[str] = ("A1", "A2", "B1", "B2", "C1", "C2")


def _load_dataframe(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["cefr"].isin(CEFR_LEVELS)]
    if df.empty:
        raise RuntimeError(
            "No labeled rows found. Expand the CEFR dictionary or regenerate labels "
            "before training."
        )
    df = df.copy()
    df["label"] = df["cefr"].apply(cefr_label_to_id)
    df["text"] = df["kaz_item"].astype(str)
    return df[["text", "label"]]


def load_dataset_splits(
    csv_path: str | Path,
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> DatasetDict:
    df = _load_dataframe(csv_path)
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
    max_length: int = 64,
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


def train_word_classifier(
    csv_path: str | Path = DEFAULT_CSV,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    epochs: int = 3,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    learning_rate: float = 3e-5,
    max_length: int = 64,
    seed: int = 42,
) -> Path:
    set_seed(seed)

    dataset = load_dataset_splits(csv_path, seed=seed)
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

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_path),
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        logging_steps=20,
        logging_dir=str(out_path / "logs"),
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
    trainer.save_model(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    print(f"Saved model and tokenizer to {out_path}")
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a word-level CEFR classifier on silver labels."
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to the silver labels CSV.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name or path to fine-tune.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Directory to store the fine-tuned model artefacts.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=16,
        help="Per-device batch size for training.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Per-device batch size for evaluation.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-5, help="Initial learning rate."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
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
    train_word_classifier(
        csv_path=args.csv,
        model_name=args.model_name,
        out_dir=args.out_dir,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
