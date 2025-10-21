from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .data import CEFR_LEVELS
from .model import RuSentenceDataset


def _load_dataset(csv_path: str | Path) -> tuple[list[str], list[list[float]]]:
    df = pd.read_csv(csv_path)
    missing = [level for level in CEFR_LEVELS if level not in df.columns]
    if missing:
        raise ValueError(f"Missing CEFR columns in dataset: {', '.join(missing)}")
    sentences = df["sentence"].astype(str).str.strip().tolist()
    probabilities = df[CEFR_LEVELS].to_numpy(dtype=np.float32).tolist()
    return sentences, probabilities


def train(
    train_csv: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "xlm-roberta-base",
    eval_csv: str | Path | None = None,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    seed: int = 42,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_sentences, train_probs = _load_dataset(train_csv)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    train_dataset = RuSentenceDataset(train_sentences, train_probs, tokenizer)

    eval_dataset = None
    if eval_csv is not None:
        eval_sentences, eval_probs = _load_dataset(eval_csv)
        eval_dataset = RuSentenceDataset(eval_sentences, eval_probs, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CEFR_LEVELS),
    )

    def compute_metrics(pred):
        probabilities = torch.softmax(torch.tensor(pred.predictions), dim=-1).numpy()
        labels = pred.label_ids
        cross_entropy = -np.mean(np.sum(labels * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1))
        predicted_levels = probabilities.argmax(axis=1)
        target_levels = labels.argmax(axis=1)
        accuracy = float((predicted_levels == target_levels).mean())
        return {"accuracy": accuracy, "cross_entropy": cross_entropy}

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        evaluation_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="cross_entropy",
        greater_is_better=False,
        seed=seed,
        report_to=[],
    )

    def collate_fn(batch: Sequence[dict]):
        labels = torch.stack([item.pop("labels") for item in batch])
        inputs = tokenizer.pad(
            batch,
            padding=True,
            return_tensors="pt",
        )
        inputs["labels"] = labels
        return inputs

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Russian CEFR sentence classifier.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to labeled training CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store the trained model.")
    parser.add_argument("--eval_csv", type=str, default=None, help="Optional evaluation CSV.")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="Base model checkpoint.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay coefficient.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for the scheduler.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        eval_csv=args.eval_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
