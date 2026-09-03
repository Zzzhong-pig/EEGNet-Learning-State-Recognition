"""Training and OOF evaluation for deployable FBCSP feature classifiers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from eeg_project.calibration import apply_temperature, fit_temperature, normalize_scores, tune_class_multipliers
from eeg_project.config import TrainConfig
from eeg_project.features import FBCSPConfig, FBCSPFeatureExtractor
from eeg_project.manifest import file_sha256
from eeg_project.metrics import bootstrap_confidence_intervals, evaluate_fold
from eeg_project.signal import filter_config_to_arrays
from eeg_project.training import data_checksum, iter_outer_splits, load_groups, set_seed


def build_fbcsp_pipeline(config: TrainConfig, estimators: int, seed: int):
    feature_config = FBCSPConfig(fs=config.filter.fs)
    extractor = FBCSPFeatureExtractor(
        fs=feature_config.fs,
        bands=feature_config.bands,
        components_per_side=feature_config.components_per_side,
        filter_order=feature_config.filter_order,
    )
    classifier = ExtraTreesClassifier(
        n_estimators=estimators,
        max_features=1.0,
        min_samples_leaf=1,
        class_weight=None,
        n_jobs=-1,
        random_state=seed,
    )
    from sklearn.pipeline import Pipeline

    return Pipeline([("fbcsp", extractor), ("classifier", classifier)])


def _cross_fitted_decision_metrics(
    probability: np.ndarray,
    truth: np.ndarray,
    classes: np.ndarray,
    config: TrainConfig,
    groups: np.ndarray | None,
) -> dict[str, Any]:
    prediction = np.empty(len(truth), dtype=np.int64)
    for _, _, train, test in iter_outer_splits(config, truth, groups):
        temperature, _ = fit_temperature(probability[train], truth[train])
        train_scaled = apply_temperature(probability[train], temperature)
        multipliers, _ = tune_class_multipliers(
            train_scaled,
            truth[train],
            target=config.selection_metric,
        )
        decision = normalize_scores(
            apply_temperature(probability[test], temperature) * multipliers.reshape(1, -1)
        )
        prediction[test] = decision.argmax(axis=1)
    metrics = evaluate_fold(truth, prediction, classes)
    return {
        "prediction": prediction,
        "metrics": {key: metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1")},
    }


def train_fbcsp_production(
    config: TrainConfig,
    output: str | Path = "artifacts/production/fbcsp",
    estimators: int = 600,
) -> dict[str, Any]:
    """Fit CV models for OOF evaluation, then one full-data deployment pipeline."""
    config.validate()
    set_seed(config.seed)
    if estimators < 1:
        raise ValueError("estimators must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x = np.load(config.data).astype(np.float32)
    labels = np.load(config.labels)
    classes, y = np.unique(labels, return_inverse=True)
    groups = load_groups(config, labels)
    if x.ndim != 3 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("Expected finite X[N,C,T] and matching labels")

    probability_sum = np.zeros((len(y), len(classes)), dtype=np.float64)
    counts = np.zeros(len(y), dtype=np.int32)
    fold_metrics = []
    for repeat, fold, train, test in iter_outer_splits(config, y, groups):
        model = build_fbcsp_pipeline(
            config,
            estimators,
            config.seed + (repeat - 1) * config.folds + fold,
        )
        model.fit(x[train], labels[train])
        probability = model.predict_proba(x[test])
        probability_sum[test] += probability
        counts[test] += 1
        predicted = probability.argmax(axis=1)
        fold_metrics.append(
            {
                "repeat": repeat,
                "fold": fold,
                **evaluate_fold(y[test], predicted, classes),
            }
        )

    if not np.all(counts == config.repeats):
        raise RuntimeError("FBCSP OOF predictions do not cover every sample")
    probability = (probability_sum / counts[:, None]).astype(np.float32)
    raw_prediction = probability.argmax(axis=1)
    cross_fitted = _cross_fitted_decision_metrics(probability, y, classes, config, groups)
    final_temperature, negative_log_likelihood = fit_temperature(probability, y)
    final_multipliers, _ = tune_class_multipliers(
        apply_temperature(probability, final_temperature),
        y,
        target=config.selection_metric,
    )

    np.savez_compressed(
        output / "oof_predictions.npz",
        sample_indices=np.arange(len(y), dtype=np.int64),
        labels=y,
        classes=classes,
        probabilities=probability,
        prediction_counts=counts,
        cross_fitted_predictions=cross_fitted["prediction"],
    )
    model = build_fbcsp_pipeline(config, estimators, config.seed)
    model.fit(x, labels)
    model_path = output / "model.joblib"
    # Compression reduces distribution size substantially; decompression happens once at startup.
    joblib.dump(model, model_path, compress=3)

    raw_metrics = evaluate_fold(y, raw_prediction, classes)
    manifest = {
        "schema_version": "1.0",
        "runtime": "sklearn_fbcsp",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_path.as_posix(),
        "model_sha256": file_sha256(model_path),
        "classes": classes.tolist(),
        "input_shape": [int(x.shape[1]), int(x.shape[2])],
        "ensemble_size": 1,
        "method": "fbcsp_extratrees",
        "selection_metric": config.selection_metric,
        "temperature": float(final_temperature),
        "class_multipliers": final_multipliers.tolist(),
        "negative_log_likelihood": float(negative_log_likelihood),
        "estimators": int(estimators),
        "feature_config": {
            "fs": config.filter.fs,
            "bands": [list(band) for band in FBCSPConfig().bands],
            "components_per_side": FBCSPConfig().components_per_side,
        },
        "filter": {
            "fs": config.filter.fs,
            "low": config.filter.low,
            "high": config.filter.high,
            "notch": config.filter.notch,
        },
        "data": {
            "samples": int(len(y)),
            "data_sha256": data_checksum(config.data),
            "labels_sha256": data_checksum(config.labels),
            "split_mode": config.split_mode,
            "groups_sha256": data_checksum(config.groups) if config.groups else None,
        },
        "oof_raw_metrics": {key: raw_metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1")},
        "oof_cross_fitted_metrics": cross_fitted["metrics"],
        "oof_raw_confidence_intervals": bootstrap_confidence_intervals(
            y,
            raw_prediction,
            config.bootstrap_iterations,
            config.seed,
        ),
        "fold_metrics": fold_metrics,
        "preprocessing": filter_config_to_arrays(config.filter),
    }
    # NumPy scalars in preprocessing cannot be serialized directly.
    manifest["preprocessing"] = {
        key: float(value) for key, value in manifest["preprocessing"].items()
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
