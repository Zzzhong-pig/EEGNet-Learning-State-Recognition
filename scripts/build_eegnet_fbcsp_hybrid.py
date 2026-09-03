"""Package a leakage-audited EEGNet + FBCSP probability fusion artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_project.hybrid import (
    cross_fitted_hybrid_metrics,
    fit_hybrid_policy,
    fuse_probabilities,
    predict_with_hybrid_policy,
)
from eeg_project.manifest import file_sha256
from eeg_project.metrics import bootstrap_confidence_intervals, evaluate_fold


def load_oof(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"sample_indices", "labels", "classes", "probabilities"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"OOF artifact {path} is missing {sorted(missing)}")
        return {name: payload[name] for name in required}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an EEGNet + FBCSP hybrid deployment manifest")
    parser.add_argument("--eegnet-artifact", default="artifacts/production/eegnet")
    parser.add_argument("--eegnet-manifest", default="artifacts/production/eegnet_ensemble/manifest.json")
    parser.add_argument("--fbcsp-artifact", default="artifacts/production/fbcsp")
    parser.add_argument("--fbcsp-manifest", default="artifacts/production/fbcsp/manifest.json")
    parser.add_argument("--output", default="artifacts/production")
    parser.add_argument("--target", choices=["accuracy", "balanced_accuracy", "macro_f1"], default="accuracy")
    parser.add_argument("--meta-folds", type=int, default=5)
    parser.add_argument("--meta-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()

    eegnet_artifact = Path(args.eegnet_artifact)
    fbcsp_artifact = Path(args.fbcsp_artifact)
    eegnet_manifest_path = Path(args.eegnet_manifest)
    fbcsp_manifest_path = Path(args.fbcsp_manifest)
    eegnet = load_oof(eegnet_artifact / "oof_predictions.npz")
    fbcsp = load_oof(fbcsp_artifact / "oof_predictions.npz")
    for name in ("sample_indices", "labels", "classes"):
        if not np.array_equal(eegnet[name], fbcsp[name]):
            raise ValueError(f"EEGNet and FBCSP OOF artifacts are not aligned: {name}")

    labels = eegnet["labels"].astype(np.int64)
    classes = eegnet["classes"]
    eegnet_probability = eegnet["probabilities"].astype(np.float32)
    fbcsp_probability = fbcsp["probabilities"].astype(np.float32)
    policy = fit_hybrid_policy(eegnet_probability, fbcsp_probability, labels, target=args.target)
    fused_probability = fuse_probabilities(
        eegnet_probability,
        fbcsp_probability,
        policy["eegnet_weight"],
    )
    full_prediction = predict_with_hybrid_policy(fused_probability, policy)
    cross_fitted = cross_fitted_hybrid_metrics(
        eegnet_probability,
        fbcsp_probability,
        labels,
        classes,
        target=args.target,
        folds=args.meta_folds,
        seed=args.meta_seed,
    )
    eegnet_manifest = json.loads(eegnet_manifest_path.read_text(encoding="utf-8"))
    fbcsp_manifest = json.loads(fbcsp_manifest_path.read_text(encoding="utf-8"))
    if eegnet_manifest.get("classes") != classes.tolist() or fbcsp_manifest.get("classes") != classes.tolist():
        raise ValueError("Deployment manifests do not match OOF class ordering")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "oof_predictions.npz",
        sample_indices=eegnet["sample_indices"],
        labels=labels,
        classes=classes,
        probabilities=fused_probability.astype(np.float32),
        cross_fitted_predictions=cross_fitted["prediction"],
    )
    full_metrics = evaluate_fold(labels, full_prediction, classes)
    input_shape = fbcsp_manifest["input_shape"]
    manifest = {
        "schema_version": "1.0",
        "runtime": "hybrid_eegnet_fbcsp",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "probability_fusion",
        "selection_metric": args.target,
        "classes": classes.tolist(),
        "input_shape": input_shape,
        "ensemble_size": int(eegnet_manifest["ensemble_size"]) + int(fbcsp_manifest["ensemble_size"]),
        "eegnet_manifest": eegnet_manifest_path.as_posix(),
        "eegnet_manifest_sha256": file_sha256(eegnet_manifest_path),
        "fbcsp_manifest": fbcsp_manifest_path.as_posix(),
        "fbcsp_manifest_sha256": file_sha256(fbcsp_manifest_path),
        "eegnet_weight": policy["eegnet_weight"],
        "fbcsp_weight": policy["fbcsp_weight"],
        "temperature": policy["temperature"],
        "class_multipliers": policy["class_multipliers"].tolist(),
        "negative_log_likelihood": policy["negative_log_likelihood"],
        "oof_full_policy_metrics": {
            key: full_metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1")
        },
        "oof_cross_fitted_metrics": cross_fitted["metrics"],
        "oof_cross_fitted_confidence_intervals": bootstrap_confidence_intervals(
            labels,
            cross_fitted["prediction"],
            args.bootstrap_iterations,
            args.meta_seed,
        ),
        "meta_validation": {
            "folds": args.meta_folds,
            "seed": args.meta_seed,
            "eegnet_weights_per_meta_fold": cross_fitted["eegnet_weights"],
            "description": "Each held-out meta fold selected fusion weight, temperature, and class multipliers from its development rows only.",
        },
        "acceptance_note": "Sample-level OOF only. Enterprise acceptance requires subject/session group IDs and group-level validation.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "artifact": str(output / "manifest.json"),
        "oof_cross_fitted_metrics": cross_fitted["metrics"],
        "eegnet_weight": policy["eegnet_weight"],
        "fbcsp_weight": policy["fbcsp_weight"],
    }, indent=2))


if __name__ == "__main__":
    main()
