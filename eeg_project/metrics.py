"""Evaluation metrics and cross-validation summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_fold(truth: np.ndarray, predicted: np.ndarray, classes: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
        "confusion_matrix": confusion_matrix(truth, predicted).tolist(),
        "classification_report": classification_report(
            truth,
            predicted,
            target_names=[str(label) for label in classes],
            output_dict=True,
            zero_division=0,
        ),
    }


def summarize_folds(fold_metrics: list[dict[str, Any]], protocol: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        key: {
            "mean": float(np.mean([fold[key] for fold in fold_metrics])),
            "std": float(np.std([fold[key] for fold in fold_metrics])),
        }
        for key in ("accuracy", "balanced_accuracy", "macro_f1")
    }
    summary["protocol"] = protocol
    summary["folds"] = fold_metrics
    return summary


def bootstrap_confidence_intervals(
    truth: np.ndarray,
    predicted: np.ndarray,
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Non-parametric 95% confidence intervals for final OOF predictions."""
    if iterations <= 0:
        return {}
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    if len(truth) != len(predicted) or len(truth) == 0:
        raise ValueError("truth and predicted must be non-empty arrays of equal length")
    rng = np.random.default_rng(seed)
    scores = {"accuracy": [], "balanced_accuracy": [], "macro_f1": []}
    for _ in range(iterations):
        sample = rng.integers(0, len(truth), len(truth))
        sample_truth = truth[sample]
        sample_predicted = predicted[sample]
        scores["accuracy"].append(float(accuracy_score(sample_truth, sample_predicted)))
        scores["balanced_accuracy"].append(float(balanced_accuracy_score(sample_truth, sample_predicted)))
        scores["macro_f1"].append(float(f1_score(sample_truth, sample_predicted, average="macro", zero_division=0)))
    return {
        name: {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
        }
        for name, values in scores.items()
    }
