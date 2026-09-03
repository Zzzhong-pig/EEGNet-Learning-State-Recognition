"""Post-hoc probability calibration for imbalanced classes."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Turn non-negative decision scores into probability distributions."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.clip(scores, 0.0, None)
    totals = scores.sum(axis=1, keepdims=True)
    fallback = totals[:, 0] <= 0.0
    if np.any(fallback):
        scores = scores.copy()
        scores[fallback] = 1.0
        totals = scores.sum(axis=1, keepdims=True)
    return (scores / totals).astype(np.float32)


def score_predictions(truth: np.ndarray, predicted: np.ndarray, target: str) -> float:
    if target == "accuracy":
        return float(accuracy_score(truth, predicted))
    if target == "balanced_accuracy":
        return float(balanced_accuracy_score(truth, predicted))
    if target == "macro_f1":
        return float(f1_score(truth, predicted, average="macro", zero_division=0))
    raise ValueError(f"Unsupported selection target: {target}")


def apply_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scale a softmax output without requiring model logits."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0)
    logits = np.log(probability) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return normalize_scores(scaled)


def fit_temperature(probability: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """Fit a single temperature on validation/OOF predictions using NLL."""
    probability = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int64)

    def objective(temperature: float) -> float:
        return float(log_loss(truth, apply_temperature(probability, temperature), labels=np.arange(probability.shape[1])))

    result = minimize_scalar(objective, bounds=(0.25, 5.0), method="bounded")
    temperature = float(result.x if result.success else 1.0)
    return temperature, objective(temperature)


def predict_with_multipliers(probability: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    adjusted = normalize_scores(probability * multipliers.reshape(1, -1))
    return adjusted.argmax(axis=1)


def tune_for_accuracy(
    probability: np.ndarray,
    truth: np.ndarray,
    grid: tuple[float, ...] = (
        0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6, 1.7, 1.8, 2.0,
    ),
) -> tuple[np.ndarray, float]:
    num_classes = probability.shape[1]
    multipliers = np.ones(num_classes, dtype=np.float32)
    best = score_predictions(truth, probability.argmax(axis=1), "accuracy")
    improved = True
    while improved:
        improved = False
        for class_index in range(num_classes):
            for value in grid:
                candidate = multipliers.copy()
                candidate[class_index] = value
                predicted = predict_with_multipliers(probability, candidate)
                score = score_predictions(truth, predicted, "accuracy")
                if score > best + 1e-6:
                    best = score
                    multipliers = candidate
                    improved = True
    return multipliers, float(best)


def ovr_predict(probability: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability)
    adjusted = probability - thresholds.reshape(1, -1)
    eligible = adjusted >= 0
    predicted = probability.argmax(axis=1)
    eligible_rows = eligible.any(axis=1)
    if np.any(eligible_rows):
        selected = adjusted[eligible_rows].copy()
        selected[~eligible[eligible_rows]] = -np.inf
        predicted[eligible_rows] = selected.argmax(axis=1)
    return predicted


def tune_ovr_thresholds(
    probability: np.ndarray,
    truth: np.ndarray,
    grid: tuple[float, ...] = (0.2, 0.25, 0.3, 0.33, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6),
    target: str = "accuracy",
) -> tuple[np.ndarray, float]:
    thresholds = np.full(probability.shape[1], 0.34, dtype=np.float32)
    best = score_predictions(truth, probability.argmax(axis=1), target)
    improved = True
    while improved:
        improved = False
        for class_index in range(probability.shape[1]):
            for value in grid:
                trial = thresholds.copy()
                trial[class_index] = value
                predicted = ovr_predict(probability, trial)
                score = score_predictions(truth, predicted, target)
                if score > best + 1e-6:
                    best = score
                    thresholds = trial
                    improved = True
    return thresholds, float(best)


def tune_class_multipliers(
    probability: np.ndarray,
    truth: np.ndarray,
    grid: tuple[float, ...] = (0.8, 1.0, 1.2, 1.5, 2.0, 2.5),
    target: str = "macro_f1",
) -> tuple[np.ndarray, float]:
    if target == "accuracy":
        return tune_for_accuracy(probability, truth, grid)
    num_classes = probability.shape[1]
    multipliers = np.ones(num_classes, dtype=np.float32)
    best = score_predictions(truth, probability.argmax(axis=1), target)
    improved = True
    while improved:
        improved = False
        for class_index in range(num_classes):
            for value in grid:
                candidate = multipliers.copy()
                candidate[class_index] = value
                predicted = predict_with_multipliers(probability, candidate)
                score = score_predictions(truth, predicted, target)
                if score > best + 1e-6:
                    best = score
                    multipliers = candidate
                    improved = True
    return multipliers, float(best)
