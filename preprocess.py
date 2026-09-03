"""EEG preprocessing CLI: validation, zero-phase filtering, and export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eeg_project.config import FilterConfig, TrainConfig
from eeg_project.logging_config import setup_logging
from eeg_project.signal import apply_filter, filter_eeg

# Re-export for backward compatibility with tests and external imports.
__all__ = ["validate_data", "filter_eeg", "preprocess_arrays"]


def validate_data(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 3:
        raise ValueError(f"X must have shape (samples, channels, time), got {x.shape}")
    if y.ndim != 1 or len(x) != len(y):
        raise ValueError("y must be one-dimensional and match X sample count")
    if not np.isfinite(x).all():
        raise ValueError("X contains NaN or infinity")
    if len(np.unique(y)) < 2:
        raise ValueError("At least two classes are required")


def preprocess_arrays(x: np.ndarray, filter_cfg: FilterConfig) -> np.ndarray:
    return apply_filter(x, filter_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess EEG arrays")
    parser.add_argument("--config", help="YAML config path (filter params read from config)")
    parser.add_argument("--input", default="data/X_features.npy")
    parser.add_argument("--labels", default="data/y_labels.npy")
    parser.add_argument("--output", default="data/X_filtered.npy")
    parser.add_argument("--fs", type=float, default=250.0)
    parser.add_argument("--low", type=float, default=4.0)
    parser.add_argument("--high", type=float, default=40.0)
    parser.add_argument("--notch", type=float, default=50.0)
    args = parser.parse_args()
    logger = setup_logging()

    if args.config:
        config = TrainConfig.from_yaml(args.config)
        filter_cfg = config.filter
        if not Path(args.input).exists() or args.input == "data/X_features.npy":
            args.input = "data/X_features.npy"
        if config.data.endswith(".npy"):
            args.output = config.data
    else:
        filter_cfg = FilterConfig(fs=args.fs, low=args.low, high=args.high, notch=args.notch)

    x = np.load(args.input).astype(np.float32)
    y = np.load(args.labels)
    validate_data(x, y)
    processed = preprocess_arrays(x, filter_cfg)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, processed)
    metadata = {
        "shape": list(processed.shape),
        "fs": filter_cfg.fs,
        "bandpass": [filter_cfg.low, filter_cfg.high],
        "notch": filter_cfg.notch,
        "normalization": "fold-wise z-score applied during training only",
        "source": str(Path(args.input).resolve()),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved %s to %s", processed.shape, output)


if __name__ == "__main__":
    main()
