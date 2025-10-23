from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from cefr.data import CEFR_LEVELS


class RuSentenceDataset(Dataset):
    def __init__(
        self,
        sentences,
        targets,
        tokenizer,
        *,
        max_length=256,
    ):
        self.sentences = sentences
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.squeeze(0) for key, value in encoded.items()}
        label = torch.tensor(self.targets[idx], dtype=torch.float32)
        inputs["labels"] = label
        return inputs


class RuSentenceCefrModel:
    __slots__ = ("tokenizer", "model", "levels")

    def __init__(self, tokenizer, model, levels=CEFR_LEVELS):
        self.tokenizer = tokenizer
        self.model = model
        self.levels = levels

    @classmethod
    def from_pretrained(cls, checkpoint, *, device=None):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=len(CEFR_LEVELS),
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_device = torch.device(device)
        model.to(torch_device)
        model.eval()
        return cls(tokenizer=tokenizer, model=model)

    @torch.inference_mode()
    def predict_proba(self, sentence):
        encoded = self.tokenizer(
            sentence,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        logits = self.model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return {level: float(prob) for level, prob in zip(self.levels, probs)}

    @torch.inference_mode()
    def predict(self, sentence):
        probs = self.predict_proba(sentence)
        level, confidence = max(probs.items(), key=lambda item: item[1])
        return level, confidence, probs


__all__ = ["RuSentenceDataset", "RuSentenceCefrModel"]
