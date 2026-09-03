"""EEG-specific batch augmentation and mixup."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


class EEGSequence(tf.keras.utils.Sequence):
    """On-the-fly augmentation for EEG tensors shaped [N, C, T, 1]."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        seed: int,
        augment: bool = True,
        mixup_alpha: float = 0.0,
        oversample: bool = False,
        minority_boost: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.oversample = oversample
        self.minority_boost = minority_boost
        self.rng = np.random.default_rng(seed)
        self.sample_weights = self._build_sample_weights()
        self.label_indices = self.y.argmax(axis=1) if self.y.ndim > 1 else self.y.astype(int)
        self.minority_classes = self._minority_classes()
        self.indices = np.arange(len(x))
        self.on_epoch_end()

    def _minority_classes(self) -> set[int]:
        counts = np.bincount(self.label_indices, minlength=int(self.label_indices.max()) + 1)
        threshold = counts.mean() * 0.6
        return {index for index, count in enumerate(counts) if count < threshold}

    def _build_sample_weights(self) -> np.ndarray:
        labels = self.y.argmax(axis=1) if self.y.ndim > 1 else self.y.astype(int)
        counts = np.bincount(labels, minlength=int(labels.max()) + 1).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        weights = 1.0 / counts[labels]
        if self.oversample:
            weights *= len(labels) / counts[labels]
        return weights / weights.sum()

    def __len__(self) -> int:
        return int(np.ceil(len(self.x) / self.batch_size))

    def _spatial_temporal_augment(self, batch: np.ndarray, batch_labels: np.ndarray) -> np.ndarray:
        scale = self.rng.uniform(0.88, 1.12, (len(batch), 1, 1, 1))
        noise = self.rng.normal(0, 0.012, batch.shape)
        batch = batch * scale + noise
        for sample_index, sample in enumerate(batch):
            label = int(batch_labels[sample_index])
            minority = self.minority_boost and label in self.minority_classes
            shift_range = 24 if minority else 16
            shift = int(self.rng.integers(-shift_range, shift_range + 1))
            if shift > 0:
                sample[:, shift:, :] = sample[:, :-shift, :]
                sample[:, :shift, :] = 0
            elif shift < 0:
                sample[:, :shift, :] = sample[:, -shift:, :]
                sample[:, shift:, :] = 0
            if self.rng.random() < (0.20 if minority else 0.12):
                sample[int(self.rng.integers(sample.shape[0]))] = 0
            if self.rng.random() < (0.14 if minority else 0.08):
                mask_len = 48 if minority else 32
                start = int(self.rng.integers(0, max(1, sample.shape[1] - mask_len)))
                sample[:, start:start + mask_len, :] = 0
            if minority and self.rng.random() < 0.25:
                sample += self.rng.normal(0, 0.02, sample.shape)
        return batch

    def _apply_mixup(self, batch_x: np.ndarray, batch_y: np.ndarray):
        if self.mixup_alpha <= 0 or len(batch_x) < 2:
            return batch_x, batch_y
        lam = self.rng.beta(self.mixup_alpha, self.mixup_alpha, size=len(batch_x))
        lam_x = lam.reshape(-1, 1, 1, 1)
        lam_y = lam.reshape(-1, 1)
        perm = self.rng.permutation(len(batch_x))
        mixed_x = batch_x * lam_x + batch_x[perm] * (1.0 - lam_x)
        mixed_y = batch_y * lam_y + batch_y[perm] * (1.0 - lam_y)
        return mixed_x.astype(np.float32), mixed_y.astype(np.float32)

    def __getitem__(self, index: int):
        ids = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x = self.x[ids].copy()
        batch_y = self.y[ids]
        if self.augment:
            batch_labels = self.label_indices[ids]
            batch_x = self._spatial_temporal_augment(batch_x, batch_labels)
            batch_x, batch_y = self._apply_mixup(batch_x, batch_y)
        return batch_x.astype(np.float32), batch_y

    def on_epoch_end(self) -> None:
        if self.oversample:
            self.indices = self.rng.choice(
                len(self.x),
                size=len(self.x),
                replace=True,
                p=self.sample_weights,
            )
        else:
            self.rng.shuffle(self.indices)
