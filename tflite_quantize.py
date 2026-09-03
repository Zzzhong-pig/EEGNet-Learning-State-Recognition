"""Export a trained Keras EEG model to deployable TFLite variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from arl_eegmodels.EEGModels import SpectralBandPower  # Registers custom spectral layer.
from eeg_project.losses import CategoricalFocalLoss, WeightedCategoricalCrossentropy
from eeg_project.manifest import build_manifest, save_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration-data", required=True)
    parser.add_argument("--preprocessing", default="")
    parser.add_argument("--output", default="artifacts/export")
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(
        args.model,
        custom_objects={
            "CategoricalFocalLoss": CategoricalFocalLoss,
            "WeightedCategoricalCrossentropy": WeightedCategoricalCrossentropy,
        },
    )
    x = np.load(args.calibration_data).astype(np.float32)
    if x.ndim == 3:
        x = x[..., None]
    x = x[: args.samples]

    variants = {}
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    variants["fp32"] = converter.convert()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    variants["fp16"] = converter.convert()

    def representative():
        for sample in x:
            yield [sample[None]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    variants["int8_dynamic_io"] = converter.convert()

    manifest = {}
    for name, content in variants.items():
        path = output / f"eegnet_{name}.tflite"
        path.write_bytes(content)
        manifest[name] = {"path": str(path), "bytes": len(content)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

    if args.preprocessing:
        deployment_manifest = build_manifest(args.model, args.preprocessing)
        deployment_manifest["tflite"] = manifest
        save_manifest(deployment_manifest, output / "deployment_manifest.json")


if __name__ == "__main__":
    main()
