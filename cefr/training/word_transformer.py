from __future__ import annotations

import json
from dataclasses import dataclass
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

CEFR_LEVELS: Sequence[str] = ("A1", "A2", "B1", "B2", "C1", "C2")


@dataclass(slots=True)
class WordTransformerConfig:
    dataset_path: Path
    output_dir: Path
    text_column: str = "rus_item"
    label_column: str = "cefr"
    model_name: str = "cointegrated/rubert-tiny2"
    test_size: float = 0.2
    random_state: int = 42
    max_length: int = 16
    epochs: int = 5
    train_batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)
        self.output_dir = Path(self.output_dir)


def _load_dataframe(config: WordTransformerConfig) -> pd.DataFrame:
    if not config.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {config.dataset_path}")
    df = pd.read_csv(config.dataset_path)
    missing = [col for col in (config.text_column, config.label_column) if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset {config.dataset_path} is missing required columns: {', '.join(missing)}"
        )
    df = df[[config.text_column, config.label_column]].dropna()
    df[config.text_column] = df[config.text_column].astype(str).str.strip()
    df = df[df[config.text_column] != ""]
    df[config.label_column] = df[config.label_column].astype(str).str.upper()
    df = df[df[config.label_column].isin(CEFR_LEVELS)]
    counts = df[config.label_column].value_counts()
    valid_labels = counts[counts >= 2].index
    df = df[df[config.label_column].isin(valid_labels)]
    if df.empty:
        raise RuntimeError("No usable rows after filtering. Check dataset contents.")
    if df[config.label_column].nunique() < 2:
        raise RuntimeError("Need at least two CEFR classes with >=2 samples to train.")
    return df.rename(columns={config.text_column: "text", config.label_column: "label"})


def _build_dataset_splits(df: pd.DataFrame, config: WordTransformerConfig) -> DatasetDict:
    train_df, val_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df["label"],
    )

    def _to_dataset(frame: pd.DataFrame) -> Dataset:
        dataset = Dataset.from_pandas(frame.reset_index(drop=True))
        drop_cols = [col for col in dataset.column_names if col not in {"text", "label"}]
        if drop_cols:
            dataset = dataset.remove_columns(drop_cols)
        return dataset

    return DatasetDict(train=_to_dataset(train_df), test=_to_dataset(val_df))


def _tokenize_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer, max_length: int) -> DatasetDict:
    def _tokenise(batch: dict[str, list[str]]) -> dict[str, list]:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        labels = batch["label"]
        if isinstance(labels[0], (list, tuple)):
            labels = [lbl[0] for lbl in labels]
        return {**enc, "labels": labels}

    tokenized = dataset.map(
        _tokenise,
        batched=True,
        remove_columns=["text", "label"],
        desc="Tokenising dataset",
    )
    tokenized.set_format(type="torch")
    return tokenized


def _compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = float((preds == labels).mean())
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def train_word_transformer(config: WordTransformerConfig) -> dict[str, object]:
    df = _load_dataframe(config)
    dataset = _build_dataset_splits(df, config)
    dataset = dataset.class_encode_column("label")
    label_list = dataset["train"].features["label"].names
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    tokenized_dataset = _tokenize_dataset(dataset, tokenizer, config.max_length)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label_list),
        label2id=label2id,
        id2label=id2label,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=20,
        logging_dir=str(config.output_dir / "logs"),
        report_to=[],
        seed=config.random_state,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))

    metrics_path = config.output_dir / "word_transformer_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "labels": label_list,
                "config": {
                    "dataset_path": str(config.dataset_path),
                    "text_column": config.text_column,
                    "label_column": config.label_column,
                    "model_name": config.model_name,
                    "test_size": config.test_size,
                    "epochs": config.epochs,
                    "train_batch_size": config.train_batch_size,
                    "eval_batch_size": config.eval_batch_size,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "warmup_ratio": config.warmup_ratio,
                    "max_length": config.max_length,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "model_dir": str(config.output_dir),
        "metrics_path": str(metrics_path),
        "accuracy": float(metrics.get("eval_accuracy", 0.0)),
        "macro_f1": float(metrics.get("eval_macro_f1", 0.0)),
        "labels": list(label_list),
    }


__all__ = ["WordTransformerConfig", "train_word_transformer"]
