"""Compact, configurable EEGNet implementation."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, constraints, layers, regularizers


@tf.keras.utils.register_keras_serializable(package="arl_eegmodels")
class SpectralBandPower(layers.Layer):
    """Log band-power features for a small, deployment-friendly EEG branch."""

    def __init__(self, fs: float, bands: list[tuple[float, float]], **kwargs):
        super().__init__(**kwargs)
        self.fs = float(fs)
        self.bands = tuple(tuple(float(value) for value in band) for band in bands)

    def call(self, inputs):
        signal = tf.squeeze(inputs, axis=-1)
        spectrum = tf.signal.rfft(signal)
        power = tf.math.square(tf.math.abs(spectrum))
        samples = tf.cast(tf.shape(signal)[-1], tf.float32)
        frequencies = tf.range(tf.shape(power)[-1], dtype=tf.float32) * self.fs / samples
        values = []
        for low, high in self.bands:
            mask = tf.cast((frequencies >= low) & (frequencies < high), power.dtype)
            denominator = tf.maximum(tf.reduce_sum(mask), tf.constant(1.0, power.dtype))
            values.append(tf.reduce_sum(power * mask[None, None, :], axis=-1) / denominator)
        return tf.math.log1p(tf.stack(values, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"fs": self.fs, "bands": [list(band) for band in self.bands]})
        return config


@tf.keras.utils.register_keras_serializable(package="arl_eegmodels")
class SpatialLogVariance(layers.Layer):
    """CSP-inspired temporal log-variance over learned EEGNet spatial filters."""

    def call(self, inputs):
        return tf.math.log(tf.math.reduce_variance(inputs, axis=2) + 1e-6)


@tf.keras.utils.register_keras_serializable(package="arl_eegmodels")
class BandSelect(layers.Layer):
    """Select one filter-bank map while preserving a Keras-serializable graph."""

    def __init__(self, index: int, **kwargs):
        super().__init__(**kwargs)
        self.index = int(index)

    def call(self, inputs):
        return inputs[..., self.index:self.index + 1]

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config


def EEGNet(
    nb_classes: int,
    Chans: int = 64,
    Samples: int = 128,
    dropoutRate: float = 0.5,
    kernLength: int = 64,
    F1: int = 8,
    D: int = 2,
    F2: int | None = None,
    norm_rate: float = 0.25,
    dropoutType: str = "Dropout",
    dense_units: int = 0,
    dense_dropout: float = 0.3,
    se_ratio: int = 0,
    weight_decay: float = 0.0,
    spectral_features: bool = False,
    spectral_bands: list[tuple[float, float]] | None = None,
    spectral_units: int = 16,
    sampling_rate: float = 250.0,
    input_bands: int = 1,
    variance_head: bool = False,
    variance_units: int = 16,
) -> Model:
    """Build EEGNet with optional squeeze-excitation and compact dense head."""
    if nb_classes < 2 or Chans < 1 or Samples < 32 or input_bands < 1:
        raise ValueError("Invalid class count or EEG input dimensions")
    if dropoutType not in {"Dropout", "SpatialDropout2D"}:
        raise ValueError("dropoutType must be Dropout or SpatialDropout2D")
    F2 = F2 or F1 * D
    dropout = layers.Dropout if dropoutType == "Dropout" else layers.SpatialDropout2D
    kernel_regularizer = regularizers.L2(weight_decay) if weight_decay else None

    inputs = layers.Input((Chans, Samples, input_bands), name="eeg")
    x = layers.Conv2D(F1, (1, kernLength), padding="same", use_bias=False,
                      kernel_regularizer=kernel_regularizer)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D,
                               depthwise_constraint=constraints.max_norm(1.0),
                               depthwise_regularizer=kernel_regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    variance_features = None
    if variance_head:
        variance_features = SpatialLogVariance(name="spatial_log_variance")(x)
        variance_features = layers.Flatten(name="spatial_variance_features")(variance_features)
        variance_features = layers.BatchNormalization(name="spatial_variance_normalization")(variance_features)
        variance_features = layers.Dense(
            variance_units,
            activation="swish",
            kernel_regularizer=kernel_regularizer,
            name="spatial_variance_projection",
        )(variance_features)
    x = layers.AveragePooling2D((1, 4))(x)
    x = dropout(dropoutRate)(x)
    x = layers.SeparableConv2D(F2, (1, 16), padding="same", use_bias=False,
                               depthwise_regularizer=kernel_regularizer,
                               pointwise_regularizer=kernel_regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = dropout(dropoutRate)(x)

    if se_ratio:
        hidden = max(F2 // se_ratio, 4)
        scale = layers.GlobalAveragePooling2D()(x)
        scale = layers.Dense(hidden, activation="swish")(scale)
        scale = layers.Dense(F2, activation="sigmoid")(scale)
        scale = layers.Reshape((1, 1, F2))(scale)
        x = layers.Multiply()([x, scale])

    x = layers.Flatten(name="features")(x)
    if spectral_features:
        bands = spectral_bands or [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 40.0)]
        spectral = SpectralBandPower(sampling_rate, bands, name="spectral_band_power")(inputs)
        spectral = layers.Flatten(name="spectral_features")(spectral)
        spectral = layers.BatchNormalization(name="spectral_normalization")(spectral)
        spectral = layers.Dense(spectral_units, activation="swish", name="spectral_projection")(spectral)
        x = layers.Concatenate(name="fused_features")([x, spectral])
    if variance_features is not None:
        x = layers.Concatenate(name="variance_fused_features")([x, variance_features])
    if dense_units:
        x = layers.Dense(dense_units, activation="swish",
                         kernel_regularizer=kernel_regularizer)(x)
        x = layers.Dropout(dense_dropout)(x)
    outputs = layers.Dense(nb_classes, activation="softmax", name="probabilities",
                           kernel_constraint=constraints.max_norm(norm_rate))(x)
    return Model(inputs, outputs, name="EEGNet")


def FilterBankEEGNet(
    nb_classes: int,
    Chans: int,
    Samples: int,
    Bands: int,
    dropoutRate: float = 0.4,
    kernLength: int = 63,
    F1: int = 8,
    D: int = 2,
    F2: int | None = None,
    dense_units: int = 64,
    dense_dropout: float = 0.3,
    se_ratio: int = 4,
    variance_units: int = 12,
    weight_decay: float = 1e-4,
) -> Model:
    """Multi-branch EEGNet that preserves one EEGNet pathway per frequency band."""
    if nb_classes < 2 or Chans < 2 or Samples < 32 or Bands < 2:
        raise ValueError("FilterBankEEGNet requires valid multi-band EEG dimensions")
    F2 = F2 or F1 * D
    regularizer = regularizers.L2(weight_decay) if weight_decay else None
    inputs = layers.Input((Chans, Samples, Bands), name="filter_bank_eeg")
    branch_features = []

    for band in range(Bands):
        prefix = f"band_{band + 1}"
        x = BandSelect(band, name=f"{prefix}_select")(inputs)
        x = layers.Conv2D(
            F1,
            (1, kernLength),
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizer,
            name=f"{prefix}_temporal",
        )(x)
        x = layers.BatchNormalization(name=f"{prefix}_temporal_bn")(x)
        x = layers.DepthwiseConv2D(
            (Chans, 1),
            use_bias=False,
            depth_multiplier=D,
            depthwise_constraint=constraints.max_norm(1.0),
            depthwise_regularizer=regularizer,
            name=f"{prefix}_spatial",
        )(x)
        x = layers.BatchNormalization(name=f"{prefix}_spatial_bn")(x)
        x = layers.Activation("elu", name=f"{prefix}_spatial_activation")(x)
        variance = SpatialLogVariance(name=f"{prefix}_log_variance")(x)
        variance = layers.Flatten(name=f"{prefix}_variance_flatten")(variance)
        variance = layers.Dense(
            variance_units,
            activation="swish",
            kernel_regularizer=regularizer,
            name=f"{prefix}_variance_projection",
        )(variance)
        x = layers.AveragePooling2D((1, 4), name=f"{prefix}_pool_1")(x)
        x = layers.Dropout(dropoutRate, name=f"{prefix}_dropout_1")(x)
        x = layers.SeparableConv2D(
            F2,
            (1, 16),
            padding="same",
            use_bias=False,
            depthwise_regularizer=regularizer,
            pointwise_regularizer=regularizer,
            name=f"{prefix}_separable",
        )(x)
        x = layers.BatchNormalization(name=f"{prefix}_separable_bn")(x)
        x = layers.Activation("elu", name=f"{prefix}_separable_activation")(x)
        if se_ratio:
            hidden = max(F2 // se_ratio, 4)
            scale = layers.GlobalAveragePooling2D(name=f"{prefix}_se_gap")(x)
            scale = layers.Dense(hidden, activation="swish", name=f"{prefix}_se_reduce")(scale)
            scale = layers.Dense(F2, activation="sigmoid", name=f"{prefix}_se_expand")(scale)
            scale = layers.Reshape((1, 1, F2), name=f"{prefix}_se_reshape")(scale)
            x = layers.Multiply(name=f"{prefix}_se_scale")([x, scale])
        x = layers.AveragePooling2D((1, 8), name=f"{prefix}_pool_2")(x)
        x = layers.Dropout(dropoutRate, name=f"{prefix}_dropout_2")(x)
        x = layers.Flatten(name=f"{prefix}_features")(x)
        branch_features.extend([x, variance])

    x = layers.Concatenate(name="filter_bank_features")(branch_features)
    x = layers.BatchNormalization(name="fusion_normalization")(x)
    x = layers.Dense(dense_units, activation="swish", kernel_regularizer=regularizer, name="fusion_dense")(x)
    x = layers.Dropout(dense_dropout, name="fusion_dropout")(x)
    outputs = layers.Dense(nb_classes, activation="softmax", name="probabilities")(x)
    return Model(inputs, outputs, name="FilterBankEEGNet")
  
