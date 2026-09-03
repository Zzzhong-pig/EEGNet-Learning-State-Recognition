"""Loss functions for imbalanced EEG classification."""

from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="eeg_project")
class CategoricalFocalLoss(tf.keras.losses.Loss):
    """Multi-class focal loss with optional per-class alpha weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: list[float] | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "sum_over_batch_size",
        name: str = "categorical_focal_loss",
        **kwargs,
    ):
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.gamma = gamma
        self.alpha = tf.constant(alpha, dtype=tf.float32) if alpha else None
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        if self.label_smoothing > 0:
            num_classes = tf.shape(y_true)[-1]
            y_true = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / tf.cast(
                num_classes, tf.float32
            )
        cross_entropy = -y_true * tf.math.log(y_pred)
        if self.alpha is not None:
            cross_entropy *= self.alpha
        modulating = tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_sum(modulating * cross_entropy, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "gamma": self.gamma,
                "alpha": self.alpha.numpy().tolist() if self.alpha is not None else None,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="eeg_project")
class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    """Class-balanced cross entropy that remains correct for Mixup labels."""

    def __init__(
        self,
        class_weights: list[float],
        label_smoothing: float = 0.0,
        reduction: str = "sum_over_batch_size",
        name: str = "weighted_categorical_crossentropy",
        **kwargs,
    ):
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        if self.label_smoothing > 0:
            classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / classes
        cross_entropy = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        sample_weight = tf.reduce_sum(y_true * self.class_weights, axis=-1)
        return cross_entropy * sample_weight

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "class_weights": self.class_weights.numpy().tolist(),
                "label_smoothing": self.label_smoothing,
            }
        )
        return config
