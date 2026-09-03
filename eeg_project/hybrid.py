"""Leakage-safe policy fitting for an EEGNet and FBCSP probability fusion."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from eeg_project.calibration import (
    apply_temperature,
    fit_temperature,
    normalize_scores,
    score_predictions,
    tune_class_multipliers,
)
from eeg_project.metrics import evaluate_fold


def fuse_probabilities(
    eegnet_probability: np.ndarray,
    fbcsp_probability: np.ndarray,
    eegnet_weight: float,
) -> np.ndarray:
    """Fuse aligned class probabilities while preserving a valid distribution."""
    if not 0.0 <= eegnet_weight <= 1.0:
        raise ValueError("eegnet_weight must be between zero and one")
    eegnet_probability = np.asarray(eegnet_probability, dtype=np.float32)
    fbcsp_probability = np.asarray(fbcsp_probability, dtype=np.float32)
    if eegnet_probability.shape != fbcsp_probability.shape or eegnet_probability.ndim != 2:
        raise ValueError("EEGNet and FBCSP probabilities must have identical [samples, classes] shapes")
    return eegnet_weight * eegnet_probability + (1.0 - eegnet_weight) * fbcsp_probability


def predict_with_hybrid_policy(probability: np.ndarray, policy: dict[str, Any]) -> np.ndarray:
    """Apply only calibration fitted from the corresponding development split."""
    scaled = apply_temperature(probability, float(policy["temperature"]))
    multipliers = np.asarray(policy["class_multipliers"], dtype=np.float32).reshape(1, -1)
    return normalize_scores(scaled * multipliers).argmax(axis=1)


def _fit_probability_policy(probability: np.ndarray, labels: np.ndarray, target: str) -> dict[str, Any]:
    temperature, negative_log_likelihood = fit_temperature(probability, labels)
    scaled = apply_temperature(probability, temperature)
    multipliers, score = tune_class_multipliers(scaled, labels, target=target)
    return {
        "temperature": float(temperature),
        "class_multipliers": multipliers.astype(np.float32),
        "negative_log_likelihood": float(negative_log_likelihood),
        "development_score": float(score),
    }


def fit_hybrid_policy(
    eegnet_probability: np.ndarray,
    fbcsp_probability: np.ndarray,
    labels: np.ndarray,
    target: str = "accuracy",
    weight_step: float = 0.025,
) -> dict[str, Any]:
    """Select the EEGNet contribution and calibration on development data only."""
    if target not in {"accuracy", "balanced_accuracy", "macro_f1"}:
        raise ValueError("Unsupported selection target")
    if not 0.0 < weight_step <= 1.0:
        raise ValueError("weight_step must be in (0, 1]")
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) != len(eegnet_probability):
        raise ValueError("labels must match probability rows")

    best: dict[str, Any] | None = None
    # Include one explicitly to avoid floating-point range endpoints changing the search.
    for weight in np.unique(np.append(np.arange(0.0, 1.0 + weight_step / 2, weight_step), 1.0)):
        probability = fuse_probabilities(eegnet_probability, fbcsp_probability, float(weight))
        policy = _fit_probability_policy(probability, labels, target)
        predicted = predict_with_hybrid_policy(probability, policy)
        score = score_predictions(labels, predicted, target)
        if best is None or score > best["development_score"] + 1e-12:
            best = {
                **policy,
                "eegnet_weight": float(weight),
                "fbcsp_weight": float(1.0 - weight),
                "development_score": float(score),
            }
    if best is None:
        raise RuntimeError("Unable to fit a hybrid policy")
    return best


def cross_fitted_hybrid_metrics(
    eegnet_probability: np.ndarray,
    fbcsp_probability: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    target: str = "accuracy",
    folds: int = 5,
    seed: int = 2026,
) -> dict[str, Any]:
    """Evaluate policy selection on held-out OOF rows, not on the fitted policy rows."""
    labels = np.asarray(labels, dtype=np.int64)
    if folds < 2:
        raise ValueError("folds must be at least two")
    prediction = np.empty(len(labels), dtype=np.int64)
    weights: list[float] = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for development, test in splitter.split(np.zeros(len(labels)), labels):
        policy = fit_hybrid_policy(
            eegnet_probability[development],
            fbcsp_probability[development],
            labels[development],
            target=target,
        )
        probability = fuse_probabilities(
            eegnet_probability[test],
            fbcsp_probability[test],
            policy["eegnet_weight"],
        )
        prediction[test] = predict_with_hybrid_policy(probability, policy)
        weights.append(float(policy["eegnet_weight"]))
    metrics = evaluate_fold(labels, prediction, np.asarray(classes))
    return {
        "prediction": prediction,
        "eegnet_weights": weights,
        "metrics": {key: metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1")},
    }
