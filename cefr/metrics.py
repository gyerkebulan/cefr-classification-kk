"""Categorical and ordinal metrics for ordered CEFR predictions."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .data import CEFR_LEVELS


def compute_cefr_metrics(y_true, y_pred, levels=CEFR_LEVELS):
    """Return class-balanced metrics plus CEFR-distance diagnostics."""
    truth, predicted = list(y_true), list(y_pred)
    levels = list(levels)
    observed = set(truth) | set(predicted)
    scored_levels = [level for level in levels if level in observed]
    if not truth or not scored_levels:
        raise ValueError("Metrics require at least one prediction and one CEFR level")
    unknown = observed - set(levels)
    if unknown:
        raise ValueError(f"Predictions contain unordered CEFR labels: {sorted(unknown)}")

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=scored_levels, zero_division=0
    )
    rank = {level: index for index, level in enumerate(levels)}
    true_rank = np.array([rank[level] for level in truth])
    pred_rank = np.array([rank[level] for level in predicted])
    distance = np.abs(true_rank - pred_rank)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1.mean()),
        "within_1": float((distance <= 1).mean()),
        "far_error_rate": float((distance > 1).mean()),
        "ordinal_mae": float(distance.mean()),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(true_rank, pred_rank, weights="quadratic")
        ),
        "per_class": {
            str(level): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, level in enumerate(scored_levels)
        },
        "confusion_matrix": confusion_matrix(
            truth, predicted, labels=scored_levels
        ).tolist(),
        "labels": [str(level) for level in scored_levels],
    }


__all__ = ["compute_cefr_metrics"]
