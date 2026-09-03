"""Build a leakage-safe mixed EEG ensemble from out-of-fold predictions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arl_eegmodels.EEGModels import SpectralBandPower  # Registers custom spectral layer.
from eeg_project.calibration import (
    apply_temperature,
    fit_temperature,
    normalize_scores,
    ovr_predict,
    score_predictions,
    tune_class_multipliers,
    tune_ovr_thresholds,
)
from eeg_project.config import TrainConfig
from eeg_project.losses import CategoricalFocalLoss, WeightedCategoricalCrossentropy
from eeg_project.manifest import file_sha256
from eeg_project.metrics import evaluate_fold
from eeg_project.training import (
    iter_outer_splits,
    load_groups,
    normalize_from_train,
)


def load_model(path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(
        path,
        custom_objects={
            "CategoricalFocalLoss": CategoricalFocalLoss,
            "WeightedCategoricalCrossentropy": WeightedCategoricalCrossentropy,
        },
    )


def load_artifact_config(root: Path, fallback: TrainConfig) -> TrainConfig:
    config_path = root / "config.json"
    if not config_path.exists():
        return fallback
    config = TrainConfig.from_mapping(json.loads(config_path.read_text(encoding="utf-8")))
    config.validate()
    return config


def model_stems(root: Path) -> list[str]:
    stems = []
    for model_path in root.glob("*fold_*.keras"):
        if model_path.name.endswith(".checkpoint.keras"):
            continue
        preprocessing = root / f"{model_path.stem}_preprocessing.npz"
        if preprocessing.is_file():
            stems.append(model_path.stem)
    if not stems:
        raise FileNotFoundError(f"No cross-validation model/preprocessing pairs found in {root}")
    return sorted(stems)


def rebuild_oof_predictions(root: Path, config: TrainConfig) -> dict[str, np.ndarray]:
    """Regenerate OOF probabilities for legacy artifacts without an OOF file."""
    x = np.load(config.data).astype(np.float32)
    labels = np.load(config.labels)
    classes, y = np.unique(labels, return_inverse=True)
    groups = load_groups(config, labels)
    x = x[..., None]
    probability_sum = np.zeros((len(y), len(classes)), dtype=np.float64)
    counts = np.zeros(len(y), dtype=np.int32)

    for repeat, fold, _, test in iter_outer_splits(config, y, groups):
        stem = f"fold_{fold}" if config.repeats == 1 else f"repeat_{repeat}_fold_{fold}"
        model_path = root / f"{stem}.keras"
        preprocessing_path = root / f"{stem}_preprocessing.npz"
        if not model_path.is_file() or not preprocessing_path.is_file():
            raise FileNotFoundError(f"Missing OOF artifact pair for {stem} in {root}")
        with np.load(preprocessing_path) as preprocessing:
            mean = preprocessing["mean"]
            std = preprocessing["std"]
        if mean.ndim == 3:
            mean, std = mean[..., None], std[..., None]
        batch = ((x[test] - mean) / std).astype(np.float32)
        probability_sum[test] += np.asarray(load_model(model_path)(batch, training=False).numpy())
        counts[test] += 1

    if not np.all(counts == config.repeats):
        raise RuntimeError(f"Could not recreate complete OOF coverage for {root}")
    payload = {
        "sample_indices": np.arange(len(y), dtype=np.int64),
        "labels": y.astype(np.int64),
        "classes": classes,
        "probabilities": (probability_sum / counts[:, None]).astype(np.float32),
        "prediction_counts": counts,
    }
    np.savez_compressed(root / "oof_predictions.npz", **payload)
    return payload


def load_oof_predictions(root: Path, fallback: TrainConfig) -> dict[str, np.ndarray]:
    path = root / "oof_predictions.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as payload:
            required = {"sample_indices", "labels", "classes", "probabilities"}
            if not required.issubset(payload.files):
                raise ValueError(f"OOF file {path} is missing required fields")
            return {name: payload[name] for name in required}
    return rebuild_oof_predictions(root, load_artifact_config(root, fallback))


def collect_members(artifact_dirs: list[str]) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for group_index, artifact_dir in enumerate(artifact_dirs):
        root = Path(artifact_dir)
        for stem in model_stems(root):
            model_path = root / f"{stem}.keras"
            preprocessing_path = root / f"{stem}_preprocessing.npz"
            members.append(
                {
                    "artifact_dir": str(root),
                    "group_index": group_index,
                    "model": str(model_path),
                    "model_sha256": file_sha256(model_path),
                    "preprocessing": str(preprocessing_path),
                    "preprocessing_sha256": file_sha256(preprocessing_path),
                }
            )
    return members


def tune_weights(probabilities: list[np.ndarray], truth: np.ndarray, target: str) -> np.ndarray:
    candidates = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
    stacked = np.stack(probabilities, axis=0)
    weights = np.ones(len(probabilities), dtype=np.float32)

    def evaluate(candidate: np.ndarray) -> float:
        probability = np.tensordot(candidate / candidate.sum(), stacked, axes=(0, 0))
        multipliers, _ = tune_class_multipliers(probability, truth, target=target)
        predicted = normalize_scores(probability * multipliers.reshape(1, -1)).argmax(axis=1)
        return score_predictions(truth, predicted, target)

    best_score = evaluate(weights)
    for _ in range(5):
        improved = False
        for index in range(len(weights)):
            for value in candidates:
                candidate = weights.copy()
                candidate[index] = value
                score = evaluate(candidate)
                if score > best_score + 1e-6:
                    best_score, weights, improved = score, candidate, True
        if not improved:
            break
    return weights / weights.sum()


def fit_policy(
    probabilities: list[np.ndarray],
    truth: np.ndarray,
    method: str,
    target: str,
) -> dict[str, Any]:
    if method == "mean":
        weights = np.full(len(probabilities), 1.0 / len(probabilities), dtype=np.float32)
    else:
        weights = tune_weights(probabilities, truth, target)
    probability = np.tensordot(weights, np.stack(probabilities, axis=0), axes=(0, 0))
    temperature, nll = fit_temperature(probability, truth)
    probability = apply_temperature(probability, temperature)
    multipliers, _ = tune_class_multipliers(probability, truth, target=target)
    decision_probability = normalize_scores(probability * multipliers.reshape(1, -1))
    thresholds = None
    if method in {"ovr", "cal_ovr"}:
        thresholds, _ = tune_ovr_thresholds(decision_probability, truth, target=target)
    return {
        "weights": weights,
        "temperature": temperature,
        "nll": nll,
        "multipliers": multipliers,
        "ovr_thresholds": thresholds,
    }


def predict_policy(probabilities: list[np.ndarray], policy: dict[str, Any]) -> np.ndarray:
    probability = np.tensordot(policy["weights"], np.stack(probabilities, axis=0), axes=(0, 0))
    probability = apply_temperature(probability, float(policy["temperature"]))
    decision_probability = normalize_scores(probability * policy["multipliers"].reshape(1, -1))
    thresholds = policy["ovr_thresholds"]
    if thresholds is not None:
        return ovr_predict(decision_probability, thresholds)
    return decision_probability.argmax(axis=1)


def cross_fitted_metrics(
    probabilities: list[np.ndarray],
    truth: np.ndarray,
    classes: np.ndarray,
    method: str,
    target: str,
    seed: int,
) -> dict[str, Any]:
    splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
    predicted = np.empty(len(truth), dtype=np.int64)
    for train, test in splitter.split(np.zeros(len(truth)), truth):
        policy = fit_policy([probability[train] for probability in probabilities], truth[train], method, target)
        predicted[test] = predict_policy([probability[test] for probability in probabilities], policy)
    metrics = evaluate_fold(truth, predicted, classes)
    return {key: metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1")}


def build_manifest(
    artifact_dirs: list[str],
    config_path: str,
    output_dir: str,
    method: str = "ovr",
    target: str = "macro_f1",
) -> dict[str, Any]:
    fallback_config = TrainConfig.from_yaml(config_path)
    fallback_config.validate()
    oof_payloads = [load_oof_predictions(Path(path), fallback_config) for path in artifact_dirs]
    reference = oof_payloads[0]
    for artifact_dir, payload in zip(artifact_dirs[1:], oof_payloads[1:]):
        for name in ("sample_indices", "labels", "classes"):
            if not np.array_equal(reference[name], payload[name]):
                raise ValueError(f"OOF alignment mismatch for {artifact_dir}: {name}")

    probabilities = [payload["probabilities"].astype(np.float32) for payload in oof_payloads]
    truth = reference["labels"].astype(np.int64)
    classes = reference["classes"]
    final_policy = fit_policy(probabilities, truth, method, target)
    cross_fitted = cross_fitted_metrics(probabilities, truth, classes, method, target, fallback_config.seed)
    members = collect_members(artifact_dirs)
    if not members:
        raise FileNotFoundError("No models found for ensemble")

    member_weights = []
    for group_index in range(len(artifact_dirs)):
        group_members = sum(member["group_index"] == group_index for member in members)
        member_weights.extend([float(final_policy["weights"][group_index] / group_members)] * group_members)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "2.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "oof_cross_fitted_ensemble",
        "artifact_dirs": artifact_dirs,
        "members": members,
        "ensemble_size": len(members),
        "method": method,
        "selection_metric": target,
        "classes": classes.tolist(),
        "model_weights": member_weights,
        "class_multipliers": final_policy["multipliers"].tolist(),
        "temperature": float(final_policy["temperature"]),
        "oof_negative_log_likelihood": float(final_policy["nll"]),
        "oof_cross_fitted_metrics": cross_fitted,
        "data_samples": int(len(truth)),
    }
    if final_policy["ovr_thresholds"] is not None:
        manifest["ovr_thresholds"] = final_policy["ovr_thresholds"].tolist()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-safe OOF ensemble")
    parser.add_argument("--config", default="configs/eegnet.yaml")
    parser.add_argument(
        "--artifact-dirs",
        default="artifacts/production/eegnet",
    )
    parser.add_argument("--output", default="artifacts/production/eegnet_ensemble")
    parser.add_argument("--method", choices=["mean", "weighted", "ovr", "cal_ovr"], default="ovr")
    parser.add_argument("--target", choices=["accuracy", "balanced_accuracy", "macro_f1"], default="macro_f1")
    args = parser.parse_args()
    dirs = [part.strip() for part in args.artifact_dirs.split(",") if part.strip()]
    manifest = build_manifest(dirs, args.config, args.output, args.method, args.target)
    print(json.dumps(
        {
            "method": manifest["method"],
            "ensemble_size": manifest["ensemble_size"],
            "oof_cross_fitted_metrics": manifest["oof_cross_fitted_metrics"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
