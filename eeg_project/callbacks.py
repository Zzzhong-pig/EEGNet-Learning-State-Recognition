"""Training callbacks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from eeg_project.calibration import predict_with_multipliers, tune_class_multipliers


class MetricCheckpoint(tf.keras.callbacks.Callback):
    """Save checkpoint by validation macro-F1 or accuracy."""

    def __init__(
        self,
        filepath: str | Path,
        x_val: np.ndarray,
        y_val: np.ndarray,
        monitor: str = "val_macro_f1",
        calibrate: bool = False,
    ):
        super().__init__()
        self.filepath = Path(filepath)
        self.x_val = x_val
        self.y_val = y_val if y_val.ndim == 1 else y_val.argmax(axis=1)
        self.monitor = monitor
        self.calibrate = calibrate
        self.best = -1.0
        self.class_multipliers = np.ones(int(np.max(self.y_val)) + 1, dtype=np.float32)

    def on_epoch_end(self, epoch: int, logs: dict | None = None):
        probability = self.model.predict(self.x_val, verbose=0)
        if self.calibrate:
            self.class_multipliers, _ = tune_class_multipliers(probability, self.y_val)
            predicted = predict_with_multipliers(probability, self.class_multipliers)
        else:
            predicted = probability.argmax(axis=1)
        if self.monitor == "val_accuracy":
            score = float(accuracy_score(self.y_val, predicted))
        else:
            score = float(f1_score(self.y_val, predicted, average="macro"))
        logs = logs or {}
        logs[self.monitor] = score
        if score > self.best:
            self.best = score
            self.model.save(self.filepath)


class MacroF1Checkpoint(MetricCheckpoint):
    def __init__(self, filepath: str | Path, x_val: np.ndarray, y_val: np.ndarray, calibrate: bool = False):
        super().__init__(filepath, x_val, y_val, monitor="val_macro_f1", calibrate=calibrate)


class AccuracyCheckpoint(MetricCheckpoint):
    def __init__(self, filepath: str | Path, x_val: np.ndarray, y_val: np.ndarray):
        super().__init__(filepath, x_val, y_val, monitor="val_accuracy", calibrate=False)
