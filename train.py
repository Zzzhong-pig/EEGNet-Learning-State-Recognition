"""Leakage-aware cross-validation training CLI."""

from __future__ import annotations

import argparse
import json

from eeg_project.config import TrainConfig
from eeg_project.training import run_cross_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEGNet with stratified cross-validation")
    parser.add_argument("--config", help="YAML config path")
    parser.add_argument("--data", default="data/X_filtered.npy")
    parser.add_argument("--labels", default="data/y_labels.npy")
    parser.add_argument("--output", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--kern-length", type=int, default=64)
    parser.add_argument("--no-se", action="store_true")
    parser.add_argument("--mixup-alpha", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        config = TrainConfig.from_yaml(args.config)
        if args.output:
            config.output = args.output
    else:
        config = TrainConfig(
            data=args.data,
            labels=args.labels,
            output=args.output or "artifacts/production/eegnet",
            folds=args.folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            learning_rate=args.learning_rate,
            mixup_alpha=args.mixup_alpha,
        )
        config.model.kern_length = args.kern_length
        if args.no_se:
            config.model.se_ratio = 0

    if args.epochs is not None:
        config.epochs = args.epochs

    summary = run_cross_validation(config)
    printable = {key: value for key, value in summary.items() if key != "folds"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
