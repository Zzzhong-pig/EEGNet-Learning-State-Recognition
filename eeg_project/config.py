"""Configuration loading for reproducible EEG experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - optional at runtime
    yaml = None


@dataclass
class FilterConfig:
    fs: float = 250.0
    low: float = 4.0
    high: float = 40.0
    notch: float | None = 50.0


@dataclass
class ModelConfig:
    f1: int = 16
    d: int = 2
    f2: int = 32
    kern_length: int = 64
    dropout_rate: float = 0.5
    se_ratio: int = 4
    dense_units: int = 32
    dense_dropout: float = 0.3
    weight_decay: float = 1e-4
    spectral_features: bool = False
    spectral_bands: list[tuple[float, float]] = field(
        default_factory=lambda: [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 40.0)]
    )
    spectral_units: int = 16
    filter_bank: bool = False
    filter_bank_bands: list[tuple[float, float]] = field(
        default_factory=lambda: [(4.0, 8.0), (8.0, 13.0), (13.0, 20.0), (20.0, 30.0), (30.0, 40.0)]
    )
    variance_head: bool = False
    variance_units: int = 16
    temporal_downsample: int = 1
    architecture: str = "eegnet"


@dataclass
class TrainConfig:
    data: str = "data/X_filtered.npy"
    labels: str = "data/y_labels.npy"
    groups: str | None = None
    output: str = "artifacts/production/eegnet"
    folds: int = 5
    repeats: int = 1
    split_mode: str = "sample"
    val_ratio: float = 0.15
    epochs: int = 120
    batch_size: int = 32
    seed: int = 42
    learning_rate: float = 1e-3
    label_smoothing: float = 0.05
    mixup_alpha: float = 0.15
    early_stopping_patience: int = 18
    reduce_lr_patience: int = 7
    loss: str = "crossentropy"
    minority_boost: bool = False
    focal_gamma: float = 2.0
    oversample: bool = False
    class_weight_power: float = 1.0
    calibrate: bool = False
    checkpoint_metric: str = "val_loss"
    selection_metric: str = "macro_f1"
    bootstrap_iterations: int = 1000
    filter: FilterConfig = field(default_factory=FilterConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    protocol_note: str = "sample-level stratified cross-validation"

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> TrainConfig:
        payload = dict(payload)
        filter_cfg = payload.pop("filter", {})
        model_cfg = payload.pop("model", {})
        if "spectral_bands" in model_cfg:
            model_cfg["spectral_bands"] = [tuple(band) for band in model_cfg["spectral_bands"]]
        if "filter_bank_bands" in model_cfg:
            model_cfg["filter_bank_bands"] = [tuple(band) for band in model_cfg["filter_bank_bands"]]
        return cls(
            filter=FilterConfig(**filter_cfg) if filter_cfg else FilterConfig(),
            model=ModelConfig(**model_cfg) if model_cfg else ModelConfig(),
            **payload,
        )

    def validate(self) -> None:
        if self.split_mode not in {"sample", "group"}:
            raise ValueError("split_mode must be 'sample' or 'group'")
        if self.split_mode == "group" and not self.groups:
            raise ValueError("groups is required when split_mode is 'group'")
        if self.folds < 2 or self.repeats < 1:
            raise ValueError("folds must be at least 2 and repeats must be positive")
        if not 0.0 < self.val_ratio < 0.5:
            raise ValueError("val_ratio must be between 0 and 0.5")
        if self.selection_metric not in {"accuracy", "balanced_accuracy", "macro_f1"}:
            raise ValueError("selection_metric must be accuracy, balanced_accuracy, or macro_f1")
        if self.bootstrap_iterations < 0:
            raise ValueError("bootstrap_iterations must be non-negative")
        for low, high in self.model.filter_bank_bands:
            if not 0 < low < high < self.filter.fs / 2:
                raise ValueError("filter_bank_bands must fall within the Nyquist range")
        if self.model.temporal_downsample < 1:
            raise ValueError("temporal_downsample must be at least 1")
        if self.filter.high >= self.filter.fs / (2 * self.model.temporal_downsample):
            raise ValueError("temporal_downsample would alias the configured high cutoff")
        if self.model.architecture not in {"eegnet", "filter_bank_eegnet"}:
            raise ValueError("architecture must be eegnet or filter_bank_eegnet")
        if self.model.architecture == "filter_bank_eegnet" and not self.model.filter_bank:
            raise ValueError("filter_bank_eegnet architecture requires filter_bank=true")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        if yaml is None:
            raise ImportError("PyYAML is required for config files: pip install pyyaml")
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(payload or {})

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")
