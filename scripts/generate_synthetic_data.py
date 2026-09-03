"""Generate synthetic EEG data for pipeline smoke tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def generate_dataset(
    samples: int = 384,
    channels: int = 5,
    timepoints: int = 1000,
    classes: int = 3,
    fs: float = 250.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(timepoints) / fs
    x = np.zeros((samples, channels, timepoints), dtype=np.float32)
    y = rng.integers(0, classes, size=samples)

    for index in range(samples):
        label = y[index]
        base_freq = 6.0 + label * 3.0
        for channel in range(channels):
            phase = rng.uniform(0, 2 * np.pi)
            signal = np.sin(2 * np.pi * base_freq * t + phase)
            signal += 0.35 * np.sin(2 * np.pi * (base_freq + 2.0) * t + phase)
            signal += rng.normal(0, 0.15, size=timepoints)
            x[index, channel] = signal.astype(np.float32)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--samples", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y = generate_dataset(samples=args.samples, seed=args.seed)
    np.save(output_dir / "X_features.npy", x)
    np.save(output_dir / "y_labels.npy", y)
    print(f"Saved synthetic dataset: X={x.shape}, y={y.shape}")


if __name__ == "__main__":
    main()
