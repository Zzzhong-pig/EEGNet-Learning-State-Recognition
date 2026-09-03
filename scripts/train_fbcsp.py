"""Train and package the deployable FBCSP + ExtraTrees EEG classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_project.classical_training import train_fbcsp_production
from eeg_project.config import TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FBCSP + ExtraTrees EEG classifier")
    parser.add_argument("--config", default="configs/fbcsp_auxiliary.yaml")
    parser.add_argument("--output", default="artifacts/production/fbcsp")
    parser.add_argument("--estimators", type=int, default=600)
    args = parser.parse_args()
    manifest = train_fbcsp_production(
        TrainConfig.from_yaml(args.config),
        output=args.output,
        estimators=args.estimators,
    )
    print(json.dumps({
        "model": manifest["model"],
        "oof_raw_metrics": manifest["oof_raw_metrics"],
        "oof_cross_fitted_metrics": manifest["oof_cross_fitted_metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
