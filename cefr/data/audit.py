"""Aggregate-only audits for CEFR label conflicts and split leakage."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

import pandas as pd

from .corpus_resources import file_sha256


def normalize(value):
    return unicodedata.normalize("NFC", str(value)).strip().casefold()


def set_digest(values):
    payload = "\n".join(sorted(map(str, values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_frame(frame, lemma_column="lemma", label_column="cefr", context_column=None):
    required = {lemma_column, label_column}
    if context_column:
        required.add(context_column)
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing audit columns: {sorted(missing)}")
    lemma = frame[lemma_column].fillna("").map(normalize)
    label = frame[label_column].fillna("").map(normalize)
    pairs = pd.DataFrame({"lemma": lemma, "label": label})
    label_sets = pairs[pairs.lemma.ne("") & pairs.label.ne("")].groupby("lemma")["label"].nunique()
    result = {
        "rows": int(len(frame)),
        "empty_lemmas": int(lemma.eq("").sum()),
        "empty_labels": int(label.eq("").sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "unique_lemmas": int(lemma[lemma.ne("")].nunique()),
        "conflicting_label_lemmas": int((label_sets > 1).sum()),
        "label_distribution": dict(sorted(Counter(label[label.ne("")]).items())),
    }
    if context_column:
        context_hashes = frame[context_column].fillna("").map(
            lambda value: hashlib.sha256(normalize(value).encode()).hexdigest()
            if normalize(value) else ""
        )
        result.update(
            empty_contexts=int(context_hashes.eq("").sum()),
            unique_context_hashes=int(context_hashes[context_hashes.ne("")].nunique()),
        )
    return result


def audit_splits(train, evaluation, lemma_column="lemma", label_column="cefr", context_column=None):
    train_audit = audit_frame(train, lemma_column, label_column, context_column)
    evaluation_audit = audit_frame(evaluation, lemma_column, label_column, context_column)
    train_lemmas = set(train[lemma_column].fillna("").map(normalize)) - {""}
    evaluation_lemmas = set(evaluation[lemma_column].fillna("").map(normalize)) - {""}
    lemma_overlap = train_lemmas & evaluation_lemmas
    result = {
        "train": train_audit,
        "evaluation": evaluation_audit,
        "normalized_lemma_overlap": len(lemma_overlap),
        "normalized_lemma_overlap_sha256": set_digest(lemma_overlap),
    }
    if context_column:
        train_context = set(train[context_column].fillna("").map(normalize)) - {""}
        evaluation_context = set(evaluation[context_column].fillna("").map(normalize)) - {""}
        context_overlap = train_context & evaluation_context
        result.update(
            normalized_context_overlap=len(context_overlap),
            normalized_context_overlap_sha256=set_digest(context_overlap),
        )
    return result


def audit_csv_files(train_path, evaluation_path, **columns):
    train_path, evaluation_path = Path(train_path), Path(evaluation_path)
    return {
        "input_sha256": {
            "train": file_sha256(train_path),
            "evaluation": file_sha256(evaluation_path),
        },
        **audit_splits(pd.read_csv(train_path), pd.read_csv(evaluation_path), **columns),
    }


def write_audit(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)


__all__ = ["audit_csv_files", "audit_frame", "audit_splits", "write_audit"]
