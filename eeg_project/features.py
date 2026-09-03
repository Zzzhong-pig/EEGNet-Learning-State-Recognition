"""Leakage-safe filter-bank common spatial pattern features for EEG."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass(frozen=True)
class FBCSPConfig:
    fs: float = 250.0
    bands: tuple[tuple[float, float], ...] = (
        (4.0, 8.0),
        (8.0, 13.0),
        (13.0, 20.0),
        (20.0, 30.0),
        (30.0, 40.0),
    )
    components_per_side: int = 2
    filter_order: int = 4


class FBCSPFeatureExtractor(BaseEstimator, TransformerMixin):
    """Fit one-vs-rest CSP filters inside each fold, then emit log-variance features."""

    def __init__(
        self,
        fs: float = 250.0,
        bands: tuple[tuple[float, float], ...] = FBCSPConfig.bands,
        components_per_side: int = 2,
        filter_order: int = 4,
    ):
        self.fs = fs
        self.bands = bands
        self.components_per_side = components_per_side
        self.filter_order = filter_order

    @staticmethod
    def _validate_x(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3 or x.shape[1] < 2 or x.shape[2] < 32:
            raise ValueError("Expected finite EEG data shaped [samples, channels, time]")
        if not np.isfinite(x).all():
            raise ValueError("EEG data contains NaN or infinity")
        return x

    def _bandpass(self, x: np.ndarray, band: tuple[float, float]) -> np.ndarray:
        low, high = (float(value) for value in band)
        if not 0 < low < high < self.fs / 2:
            raise ValueError(f"Invalid FBCSP band: {band}")
        sos = butter(self.filter_order, [low, high], btype="bandpass", fs=self.fs, output="sos")
        return sosfiltfilt(sos, x, axis=-1).astype(np.float32)

    @staticmethod
    def _covariance(x: np.ndarray) -> np.ndarray:
        centered = x - x.mean(axis=-1, keepdims=True)
        covariance = np.einsum("nct,ndt->ncd", centered, centered) / max(x.shape[-1] - 1, 1)
        trace = np.trace(covariance, axis1=1, axis2=2)
        return covariance / (trace[:, None, None] + 1e-8)

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = self._validate_x(x)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(x):
            raise ValueError("y must be one-dimensional and match x")
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("FBCSP requires at least two classes")
        self.channels_ = x.shape[1]
        components = min(int(self.components_per_side), self.channels_ // 2)
        if components < 1:
            raise ValueError("Not enough channels for the requested CSP components")
        self.components_per_side_ = components
        self.spatial_filters_: list[list[np.ndarray]] = []

        for band in self.bands:
            covariance = self._covariance(self._bandpass(x, band))
            band_filters = []
            for label in self.classes_:
                positive = covariance[y == label]
                negative = covariance[y != label]
                if not len(positive) or not len(negative):
                    raise ValueError(f"Unable to fit one-vs-rest CSP for class {label!r}")
                class_covariance = positive.mean(axis=0)
                rest_covariance = negative.mean(axis=0)
                _, vectors = eigh(
                    class_covariance + np.eye(self.channels_) * 1e-6,
                    rest_covariance + np.eye(self.channels_) * 1e-6,
                )
                indices = np.r_[
                    np.arange(components),
                    np.arange(self.channels_ - components, self.channels_),
                ]
                band_filters.append(vectors[:, indices].astype(np.float32))
            self.spatial_filters_.append(band_filters)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "spatial_filters_"):
            raise RuntimeError("FBCSPFeatureExtractor must be fitted before transform")
        x = self._validate_x(x)
        if x.shape[1] != self.channels_:
            raise ValueError(f"Expected {self.channels_} channels, got {x.shape[1]}")
        features = []
        for band, filters in zip(self.bands, self.spatial_filters_):
            filtered = self._bandpass(x, band)
            for spatial_filter in filters:
                projected = np.einsum("cf,nct->nft", spatial_filter, filtered)
                variance = np.var(projected, axis=-1)
                normalized = variance / (variance.sum(axis=1, keepdims=True) + 1e-8)
                features.append(np.log(normalized + 1e-8))
        return np.concatenate(features, axis=1).astype(np.float32)
