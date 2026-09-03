"""Reusable prediction service for Keras EEG models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf

from arl_eegmodels.EEGModels import SpectralBandPower  # Registers custom spectral layer.
from eeg_project.calibration import apply_temperature, normalize_scores, ovr_predict
from eeg_project.losses import CategoricalFocalLoss, WeightedCategoricalCrossentropy
from eeg_project.manifest import file_sha256
from eeg_project.config import FilterConfig
from eeg_project.signal import apply_filter, apply_filter_bank, load_filter_bank_bands, load_filter_config


def _portable_artifact_path(path: str | Path) -> Path:
    """Read relative manifest paths produced on either Windows or POSIX hosts."""
    return Path(str(path).replace("\\", "/"))


class EEGPredictor:
    def __init__(self, model_path: str, preprocessing_path: str, apply_filtering: bool = True):
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "CategoricalFocalLoss": CategoricalFocalLoss,
                "WeightedCategoricalCrossentropy": WeightedCategoricalCrossentropy,
            },
        )
        data = np.load(preprocessing_path, allow_pickle=True)
        self.mean = data["mean"]
        self.std = data["std"]
        self.classes = data["classes"]
        self.class_multipliers = self._load_multipliers(data)
        self.filter_cfg = load_filter_config(data)
        self.filter_bank_bands = load_filter_bank_bands(data)
        self.raw_channels = int(data["raw_channels"]) if "raw_channels" in data else self.model.input_shape[1]
        self.raw_samples = int(data["raw_samples"]) if "raw_samples" in data else self.model.input_shape[2]
        self.temporal_downsample = int(data["temporal_downsample"]) if "temporal_downsample" in data else 1
        self.apply_filtering = apply_filtering and self.filter_cfg is not None

    def _load_multipliers(self, data) -> np.ndarray:
        if "class_multipliers" in data:
            multipliers = np.asarray(data["class_multipliers"], dtype=np.float32).reshape(-1)
            if len(multipliers) == len(self.classes):
                return multipliers
        return np.ones(len(self.classes), dtype=np.float32)

    def _validate_raw_samples(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim == 2:
            x = x[None]
        expected = tuple(self.model.input_shape[1:])
        if x.ndim != 3 or x.shape[1] != self.raw_channels or x.shape[2] != self.raw_samples:
            raise ValueError(f"Expected [batch, {self.raw_channels}, {self.raw_samples}], got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("Input contains NaN or infinity")
        return x

    def _prepare(self, samples: np.ndarray, already_filtered: bool = False) -> np.ndarray:
        x = self._validate_raw_samples(samples)
        expected = tuple(self.model.input_shape[1:])
        if self.apply_filtering and self.filter_cfg is not None and not already_filtered:
            x = apply_filter(x, self.filter_cfg)
        if self.filter_bank_bands:
            fs = self.filter_cfg.fs if self.filter_cfg is not None else 250.0
            x = apply_filter_bank(x, fs, self.filter_bank_bands)
        elif x.ndim == 3:
            x = x[..., None]
        if self.temporal_downsample > 1:
            x = x[:, :, ::self.temporal_downsample, :]
        if tuple(x.shape[1:]) != expected:
            raise ValueError(f"Prepared input shape {x.shape[1:]} does not match model shape {expected}")
        mean = self.mean
        std = self.std
        if mean.ndim == 3:
            mean = mean[..., None]
            std = std[..., None]
        return ((x - mean) / std).astype(np.float32)

    @staticmethod
    def _predict_model(model: tf.keras.Model, batch: np.ndarray) -> np.ndarray:
        return np.asarray(model(batch, training=False).numpy(), dtype=np.float32)

    def _decode(self, probability: np.ndarray) -> list[dict]:
        adjusted = normalize_scores(probability * self.class_multipliers.reshape(1, -1))
        results = []
        for raw, calibrated in zip(probability, adjusted):
            label_index = int(np.argmax(calibrated))
            results.append(
                {
                    "label": self.classes[label_index].item(),
                    "confidence": float(calibrated[label_index]),
                    "probabilities": raw.tolist(),
                    "calibrated_probabilities": calibrated.tolist(),
                }
            )
        return results

    def predict(self, samples, tta: bool = False) -> list[dict]:
        x = self._prepare(samples)
        if not tta:
            probability = self._predict_model(self.model, x)
        else:
            variants = [x, np.flip(x, axis=2)]
            probability = np.mean([self._predict_model(self.model, variant) for variant in variants], axis=0)
        return self._decode(probability)


class EnsembleEEGPredictor:
    """Average predictions across multiple fold checkpoints."""

    def __init__(self, artifact_dir: str, folds: int = 5):
        root = Path(artifact_dir)
        self.predictors = [
            EEGPredictor(str(root / f"fold_{index}.keras"), str(root / f"fold_{index}_preprocessing.npz"))
            for index in range(1, folds + 1)
            if (root / f"fold_{index}.keras").exists()
        ]
        if not self.predictors:
            raise FileNotFoundError(f"No fold models found in {artifact_dir}")

    def predict(self, samples, tta: bool = False) -> list[dict]:
        batch_probs = [predictor.predict(samples, tta=tta) for predictor in self.predictors]
        merged = []
        for index in range(len(batch_probs[0])):
            avg_prob = np.mean(
                [fold[index]["probabilities"] for fold in batch_probs],
                axis=0,
            )
            avg_multipliers = np.mean(
                [predictor.class_multipliers for predictor in self.predictors],
                axis=0,
            )
            calibrated = normalize_scores((avg_prob * avg_multipliers).reshape(1, -1))[0]
            label_index = int(np.argmax(calibrated))
            merged.append(
                {
                    "label": self.predictors[0].classes[label_index].item(),
                    "confidence": float(calibrated[label_index]),
                    "probabilities": avg_prob.tolist(),
                    "calibrated_probabilities": calibrated.tolist(),
                    "ensemble_size": len(self.predictors),
                }
            )
        return merged


class MixedEnsembleEEGPredictor:
    """Weighted multi-model ensemble with calibration and OVR thresholds."""

    def __init__(self, manifest_path: str, verify_integrity: bool = True):
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.classes = np.asarray(manifest["classes"])
        self.class_multipliers = np.asarray(manifest["class_multipliers"], dtype=np.float32)
        self.members = manifest["members"]
        self.method = manifest.get("method", "mean")
        self.temperature = float(manifest.get("temperature", 1.0))
        self.model_weights = np.asarray(manifest.get("model_weights") or [1.0] * len(self.members), dtype=np.float32)
        self.model_weights = self.model_weights / self.model_weights.sum()
        self.ovr_thresholds = np.asarray(manifest.get("ovr_thresholds") or [], dtype=np.float32)
        self.stacker = None
        if self.method == "stack" and "stacker" in manifest:
            import joblib

            self.stacker = joblib.load(manifest["stacker"])
        self.models: list[tuple[tf.keras.Model, EEGPredictor]] = []
        for member in self.members:
            self._verify_member(member, verify_integrity)
            predictor = EEGPredictor(
                str(_portable_artifact_path(member["model"])),
                str(_portable_artifact_path(member["preprocessing"])),
            )
            self.models.append((predictor.model, predictor))
        if not self.models:
            raise FileNotFoundError("No models found in mixed ensemble manifest")
        self.ensemble_size = len(self.models)
        first = self.models[0][1]
        self.input_shape = tuple(first.model.input_shape[1:3])
        self.shared_filter_cfg = first.filter_cfg
        self.shared_filtering = all(
            predictor.apply_filtering
            and predictor.filter_cfg == self.shared_filter_cfg
            and tuple(predictor.model.input_shape[1:3]) == self.input_shape
            and np.array_equal(predictor.classes, self.classes)
            for _, predictor in self.models
        )

    @staticmethod
    def _verify_member(member: dict[str, Any], verify_integrity: bool) -> None:
        model_path = _portable_artifact_path(member["model"])
        preprocessing_path = _portable_artifact_path(member["preprocessing"])
        if not model_path.is_file() or not preprocessing_path.is_file():
            raise FileNotFoundError(f"Missing ensemble artifact: {model_path} or {preprocessing_path}")
        if not verify_integrity:
            return
        for path, field in ((model_path, "model_sha256"), (preprocessing_path, "preprocessing_sha256")):
            expected = member.get(field)
            if expected and file_sha256(path) != expected:
                raise ValueError(f"Artifact integrity check failed for {path}")

    def _combine(self, probabilities: list[np.ndarray]) -> np.ndarray:
        stacked = np.stack(probabilities, axis=0)
        if self.method == "stack" and self.stacker is not None:
            return self.stacker.predict_proba(np.concatenate(probabilities, axis=1))
        if self.method in {"weighted", "ovr", "cal_ovr"}:
            return np.tensordot(self.model_weights, stacked, axes=(0, 0))
        return stacked.mean(axis=0)

    def predict_proba(self, samples, tta: bool = False) -> np.ndarray:
        """Return uncalibrated ensemble probabilities for a higher-level fusion."""
        raw = self.models[0][1]._validate_raw_samples(samples)
        filtered = apply_filter(raw, self.shared_filter_cfg) if self.shared_filtering else raw
        probabilities = []
        for model, predictor in self.models:
            batch = predictor._prepare(filtered, already_filtered=self.shared_filtering)
            if not tta:
                probabilities.append(EEGPredictor._predict_model(model, batch))
            else:
                flipped = np.flip(batch, axis=2)
                probabilities.append(
                    (EEGPredictor._predict_model(model, batch) + EEGPredictor._predict_model(model, flipped)) / 2
                )
        return self._combine(probabilities)

    def predict(self, samples, tta: bool = False) -> list[dict]:
        avg_prob = self.predict_proba(samples, tta=tta)
        if self.method == "stack" and self.stacker is not None:
            calibrated = avg_prob
            results = []
            for raw in avg_prob:
                label_index = int(np.argmax(raw))
                results.append(
                    {
                        "label": self.classes[label_index].item(),
                        "confidence": float(raw[label_index]),
                        "probabilities": raw.tolist(),
                        "calibrated_probabilities": raw.tolist(),
                        "ensemble_size": len(self.models),
                    }
                )
            return results
        avg_prob = apply_temperature(avg_prob, self.temperature)
        calibrated = normalize_scores(avg_prob * self.class_multipliers.reshape(1, -1))
        if self.method in {"ovr", "cal_ovr"} and len(self.ovr_thresholds) == len(self.classes):
            predicted_indices = ovr_predict(calibrated, self.ovr_thresholds)
            results = []
            for raw, adjusted, label_index in zip(avg_prob, calibrated, predicted_indices):
                results.append(
                    {
                        "label": self.classes[int(label_index)].item(),
                        "confidence": float(adjusted[int(label_index)]),
                        "probabilities": raw.tolist(),
                        "calibrated_probabilities": adjusted.tolist(),
                        "ensemble_size": len(self.models),
                    }
                )
            return results
        results = []
        for raw, adjusted in zip(avg_prob, calibrated):
            label_index = int(np.argmax(adjusted))
            results.append(
                {
                    "label": self.classes[label_index].item(),
                    "confidence": float(adjusted[label_index]),
                    "probabilities": raw.tolist(),
                    "calibrated_probabilities": adjusted.tolist(),
                    "ensemble_size": len(self.models),
                }
            )
        return results


class FBCSPExtraTreesPredictor:
    """Inference wrapper for a full-data FBCSP + ExtraTrees production artifact."""

    def __init__(self, manifest_path: str, verify_integrity: bool = True):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("runtime") != "sklearn_fbcsp":
            raise ValueError("Manifest is not an sklearn_fbcsp deployment artifact")
        self.model_path = _portable_artifact_path(self.manifest["model"])
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Missing FBCSP model artifact: {self.model_path}")
        expected_hash = self.manifest.get("model_sha256")
        if verify_integrity and expected_hash and file_sha256(self.model_path) != expected_hash:
            raise ValueError(f"Artifact integrity check failed for {self.model_path}")
        self.model = joblib.load(self.model_path)
        self.classes = np.asarray(self.manifest["classes"])
        self.input_shape = tuple(self.manifest["input_shape"])
        self.filter_cfg = FilterConfig(**self.manifest["filter"])
        self.class_multipliers = np.asarray(self.manifest.get("class_multipliers"), dtype=np.float32)
        if len(self.class_multipliers) != len(self.classes):
            self.class_multipliers = np.ones(len(self.classes), dtype=np.float32)
        self.temperature = float(self.manifest.get("temperature", 1.0))
        self.method = str(self.manifest.get("method", "fbcsp_extratrees"))
        self.ensemble_size = int(self.manifest.get("ensemble_size", 1))
        if not np.array_equal(np.asarray(self.model.classes_), self.classes):
            raise ValueError("FBCSP artifact classes do not match its manifest")

    def _prepare(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim == 2:
            x = x[None]
        if x.ndim != 3 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Expected [batch, {self.input_shape[0]}, {self.input_shape[1]}], got {x.shape}"
            )
        if not np.isfinite(x).all():
            raise ValueError("Input contains NaN or infinity")
        return apply_filter(x, self.filter_cfg)

    def predict_proba(self, samples) -> np.ndarray:
        """Return raw classifier probabilities before deployment calibration."""
        return np.asarray(self.model.predict_proba(self._prepare(samples)), dtype=np.float32)

    def predict(self, samples, tta: bool = False) -> list[dict]:
        if tta:
            raise ValueError("TTA is not supported by the FBCSP deployment model")
        probability = self.predict_proba(samples)
        probability = apply_temperature(probability, self.temperature)
        decision = normalize_scores(probability * self.class_multipliers.reshape(1, -1))
        results = []
        for raw, adjusted in zip(probability, decision):
            index = int(np.argmax(adjusted))
            results.append(
                {
                    "label": self.classes[index].item(),
                    "confidence": float(adjusted[index]),
                    "probabilities": raw.tolist(),
                    "calibrated_probabilities": adjusted.tolist(),
                    "ensemble_size": self.ensemble_size,
                }
            )
        return results


class HybridEEGNetFBCSPPredictor:
    """Auditable probability fusion with EEGNet as a required neural component."""

    def __init__(self, manifest_path: str, verify_integrity: bool = True):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("runtime") != "hybrid_eegnet_fbcsp":
            raise ValueError("Manifest is not an EEGNet + FBCSP hybrid deployment artifact")
        self.eegnet_manifest_path = self._resolve_reference("eegnet_manifest")
        self.fbcsp_manifest_path = self._resolve_reference("fbcsp_manifest")
        self._verify_reference(self.eegnet_manifest_path, "eegnet_manifest_sha256", verify_integrity)
        self._verify_reference(self.fbcsp_manifest_path, "fbcsp_manifest_sha256", verify_integrity)
        self.eegnet = MixedEnsembleEEGPredictor(str(self.eegnet_manifest_path), verify_integrity=verify_integrity)
        self.fbcsp = FBCSPExtraTreesPredictor(str(self.fbcsp_manifest_path), verify_integrity=verify_integrity)
        self.classes = np.asarray(self.manifest["classes"])
        if not np.array_equal(self.classes, self.eegnet.classes) or not np.array_equal(self.classes, self.fbcsp.classes):
            raise ValueError("Hybrid member class definitions do not match")
        self.input_shape = tuple(self.manifest["input_shape"])
        if self.input_shape != self.eegnet.input_shape or self.input_shape != self.fbcsp.input_shape:
            raise ValueError("Hybrid member input shapes do not match the manifest")
        self.eegnet_weight = float(self.manifest["eegnet_weight"])
        self.fbcsp_weight = float(self.manifest["fbcsp_weight"])
        if not np.isclose(self.eegnet_weight + self.fbcsp_weight, 1.0, atol=1e-6):
            raise ValueError("Hybrid weights must sum to one")
        self.temperature = float(self.manifest.get("temperature", 1.0))
        self.class_multipliers = np.asarray(self.manifest.get("class_multipliers"), dtype=np.float32)
        if len(self.class_multipliers) != len(self.classes):
            raise ValueError("Hybrid class multipliers do not match classes")
        self.method = str(self.manifest.get("method", "probability_fusion"))
        self.ensemble_size = self.eegnet.ensemble_size + self.fbcsp.ensemble_size

    def _resolve_reference(self, field: str) -> Path:
        reference = _portable_artifact_path(self.manifest[field])
        if reference.is_file():
            return reference
        local = self.manifest_path.parent / reference
        if local.is_file():
            return local
        raise FileNotFoundError(f"Missing hybrid reference {field}: {reference}")

    def _verify_reference(self, path: Path, field: str, verify_integrity: bool) -> None:
        expected = self.manifest.get(field)
        if verify_integrity and expected and file_sha256(path) != expected:
            raise ValueError(f"Artifact integrity check failed for {path}")

    def predict_proba(self, samples, tta: bool = False) -> np.ndarray:
        if tta:
            raise ValueError("TTA is not supported by the EEGNet + FBCSP hybrid")
        eegnet_probability = self.eegnet.predict_proba(samples)
        fbcsp_probability = self.fbcsp.predict_proba(samples)
        return self.eegnet_weight * eegnet_probability + self.fbcsp_weight * fbcsp_probability

    def predict(self, samples, tta: bool = False) -> list[dict]:
        probability = apply_temperature(self.predict_proba(samples, tta=tta), self.temperature)
        calibrated = normalize_scores(probability * self.class_multipliers.reshape(1, -1))
        results = []
        for raw, adjusted in zip(probability, calibrated):
            index = int(np.argmax(adjusted))
            results.append(
                {
                    "label": self.classes[index].item(),
                    "confidence": float(adjusted[index]),
                    "probabilities": raw.tolist(),
                    "calibrated_probabilities": adjusted.tolist(),
                    "ensemble_size": self.ensemble_size,
                }
            )
        return results


def load_deployment_predictor(manifest_path: str, verify_integrity: bool = True):
    """Load the predictor type declared by a deployment manifest."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("runtime") == "hybrid_eegnet_fbcsp":
        return HybridEEGNetFBCSPPredictor(manifest_path, verify_integrity=verify_integrity)
    if manifest.get("runtime") == "sklearn_fbcsp":
        return FBCSPExtraTreesPredictor(manifest_path, verify_integrity=verify_integrity)
    return MixedEnsembleEEGPredictor(manifest_path, verify_integrity=verify_integrity)


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))
