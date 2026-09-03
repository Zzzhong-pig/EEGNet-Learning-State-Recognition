"""Shared EEG signal processing for training and inference."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from eeg_project.config import FilterConfig


DEFAULT_FILTER_BANK_BANDS: tuple[tuple[float, float], ...] = (
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 20.0),
    (20.0, 30.0),
    (30.0, 40.0),
)


def filter_eeg(
    x: np.ndarray,
    fs: float,
    low: float,
    high: float,
    notch: float | None,
) -> np.ndarray:
    if not 0 < low < high < fs / 2:
        raise ValueError("Filter frequencies must satisfy 0 < low < high < Nyquist")
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    result = sosfiltfilt(sos, x, axis=-1)
    if notch and notch < fs / 2:
        b, a = iirnotch(notch, 30.0, fs)
        result = filtfilt(b, a, result, axis=-1)
    return result.astype(np.float32)


def apply_filter(x: np.ndarray, filter_cfg: FilterConfig) -> np.ndarray:
    return filter_eeg(x, filter_cfg.fs, filter_cfg.low, filter_cfg.high, filter_cfg.notch)


def apply_filter_bank(
    x: np.ndarray,
    fs: float,
    bands: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    stack_bands: bool = True,
) -> np.ndarray:
    """Create independently band-passed EEG copies as channels or input feature maps."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("Filter-bank input must have shape [samples, channels, time]")
    bands = tuple((float(low), float(high)) for low, high in bands)
    if not bands:
        raise ValueError("At least one filter-bank band is required")
    outputs = [filter_eeg(x, fs, low, high, None) for low, high in bands]
    if stack_bands:
        return np.stack(outputs, axis=-1).astype(np.float32)
    return np.concatenate(outputs, axis=1).astype(np.float32)


def filter_config_to_arrays(filter_cfg: FilterConfig) -> dict[str, np.ndarray]:
    return {
        "filter_fs": np.float32(filter_cfg.fs),
        "filter_low": np.float32(filter_cfg.low),
        "filter_high": np.float32(filter_cfg.high),
        "filter_notch": np.float32(filter_cfg.notch if filter_cfg.notch is not None else -1.0),
    }


def load_filter_config(data) -> FilterConfig | None:
    if "filter_fs" not in data:
        return None
    notch = float(data["filter_notch"])
    return FilterConfig(
        fs=float(data["filter_fs"]),
        low=float(data["filter_low"]),
        high=float(data["filter_high"]),
        notch=None if notch < 0 else notch,
    )


def load_filter_bank_bands(data) -> tuple[tuple[float, float], ...]:
    if "filter_bank_bands" not in data:
        return ()
    bands = np.asarray(data["filter_bank_bands"], dtype=np.float32)
    if bands.ndim != 2 or bands.shape[1] != 2:
        raise ValueError("Invalid filter-bank metadata")
    return tuple((float(low), float(high)) for low, high in bands)
