"""Leakage-aware cross-validation training loop."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from arl_eegmodels.EEGModels import EEGNet, FilterBankEEGNet
from eeg_project.augment import EEGSequence
from eeg_project.calibration import normalize_scores, predict_with_multipliers, tune_class_multipliers
from eeg_project.callbacks import AccuracyCheckpoint, MacroF1Checkpoint
from eeg_project.config import TrainConfig
from eeg_project.logging_config import log_event, setup_logging
from eeg_project.losses import CategoricalFocalLoss, WeightedCategoricalCrossentropy
from eeg_project.manifest import build_manifest, save_manifest
from eeg_project.metrics import bootstrap_confidence_intervals, evaluate_fold, summarize_folds
from eeg_project.signal import apply_filter_bank, filter_config_to_arrays


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def data_checksum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_from_train(train: np.ndarray, *others: np.ndarray):
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True).clip(1e-6)
    normalized = tuple(((part - mean) / std).astype(np.float32) for part in (train, *others))
    return normalized, mean, std


def save_preprocessing_bundle(
    path: Path,
    mean: np.ndarray,
    std: np.ndarray,
    classes: np.ndarray,
    config: TrainConfig,
    class_multipliers: np.ndarray,
    raw_channels: int,
    raw_samples: int,
) -> None:
    filter_bank_bands = (
        np.asarray(config.model.filter_bank_bands, dtype=np.float32)
        if config.model.filter_bank
        else np.empty((0, 2), dtype=np.float32)
    )
    np.savez(
        path,
        mean=mean,
        std=std,
        classes=classes,
        class_multipliers=class_multipliers,
        raw_channels=np.int32(raw_channels),
        raw_samples=np.int32(raw_samples),
        temporal_downsample=np.int32(config.model.temporal_downsample),
        filter_bank_bands=filter_bank_bands,
        **filter_config_to_arrays(config.filter),
    )


def transform_model_input(x: np.ndarray, config: TrainConfig) -> np.ndarray:
    if config.model.filter_bank:
        x = apply_filter_bank(x, config.filter.fs, config.model.filter_bank_bands)
    factor = config.model.temporal_downsample
    return x[:, :, ::factor] if x.ndim == 3 else x[:, :, ::factor, :]


def build_model(config: TrainConfig, num_classes: int, chans: int, samples: int) -> tf.keras.Model:
    model_cfg = config.model
    if model_cfg.architecture == "filter_bank_eegnet":
        return FilterBankEEGNet(
            num_classes,
            chans,
            samples,
            Bands=len(model_cfg.filter_bank_bands),
            dropoutRate=model_cfg.dropout_rate,
            kernLength=model_cfg.kern_length,
            F1=model_cfg.f1,
            D=model_cfg.d,
            F2=model_cfg.f2,
            dense_units=model_cfg.dense_units,
            dense_dropout=model_cfg.dense_dropout,
            se_ratio=model_cfg.se_ratio,
            variance_units=model_cfg.variance_units,
            weight_decay=model_cfg.weight_decay,
        )
    return EEGNet(
        num_classes,
        chans,
        samples,
        dropoutRate=model_cfg.dropout_rate,
        kernLength=model_cfg.kern_length,
        F1=model_cfg.f1,
        D=model_cfg.d,
        F2=model_cfg.f2,
        se_ratio=model_cfg.se_ratio,
        dense_units=model_cfg.dense_units,
        dense_dropout=model_cfg.dense_dropout,
        weight_decay=model_cfg.weight_decay,
        spectral_features=model_cfg.spectral_features,
        spectral_bands=model_cfg.spectral_bands,
        spectral_units=model_cfg.spectral_units,
        sampling_rate=config.filter.fs,
        input_bands=1 if not model_cfg.filter_bank else len(model_cfg.filter_bank_bands),
        variance_head=model_cfg.variance_head,
        variance_units=model_cfg.variance_units,
    )


def build_class_weights(y_train: np.ndarray, power: float) -> dict[int, float]:
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    if power != 1.0:
        weights = np.power(weights, power)
    weights = weights / weights.mean()
    return dict(enumerate(weights))


def build_loss(config: TrainConfig, y_train: np.ndarray):
    if config.loss == "focal":
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        if config.class_weight_power != 1.0:
            class_weights = np.power(class_weights, config.class_weight_power)
        class_weights = (class_weights / class_weights.mean()).tolist()
        return CategoricalFocalLoss(
            gamma=config.focal_gamma,
            alpha=class_weights,
            label_smoothing=config.label_smoothing,
        )
    class_weights = build_class_weights(y_train, config.class_weight_power)
    return WeightedCategoricalCrossentropy(
        class_weights=[class_weights[index] for index in range(len(class_weights))],
        label_smoothing=config.label_smoothing,
    )


def compile_model(model: tf.keras.Model, config: TrainConfig, y_train: np.ndarray) -> None:
    optimizer = tf.keras.optimizers.AdamW(
        config.learning_rate,
        weight_decay=config.model.weight_decay,
        clipnorm=1.0,
    )
    model.compile(optimizer=optimizer, loss=build_loss(config, y_train), metrics=["accuracy"])


def load_groups(config: TrainConfig, labels: np.ndarray) -> np.ndarray | None:
    if config.split_mode != "group":
        return None
    groups = np.load(config.groups, allow_pickle=False)
    if groups.ndim != 1 or len(groups) != len(labels):
        raise ValueError("groups must be a one-dimensional array matching the number of labels")
    if len(np.unique(groups)) < config.folds:
        raise ValueError("groups must contain at least as many unique values as folds")
    for label in np.unique(labels):
        if len(np.unique(groups[labels == label])) < config.folds:
            raise ValueError(
                f"Class {label!r} appears in fewer than {config.folds} groups; group CV is not possible"
            )
    return groups


def iter_outer_splits(
    config: TrainConfig,
    y: np.ndarray,
    groups: np.ndarray | None,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield repeat, fold, development, and test indices without group overlap."""
    positions = np.arange(len(y))
    for repeat in range(1, config.repeats + 1):
        seed = config.seed + repeat - 1
        if groups is None:
            splitter = StratifiedKFold(config.folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(positions, y)
        else:
            splitter = StratifiedGroupKFold(config.folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(positions, y, groups)
        for fold, (development, test) in enumerate(split_iter, 1):
            if groups is not None and np.intersect1d(groups[development], groups[test]).size:
                raise RuntimeError("Group leakage detected in cross-validation split")
            yield repeat, fold, development, test


def split_train_validation(
    development: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    config: TrainConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if groups is None:
        return train_test_split(
            development,
            test_size=config.val_ratio,
            stratify=y[development],
            random_state=seed,
        )

    group_values = groups[development]
    unique_groups = np.unique(group_values)
    inner_folds = min(5, len(unique_groups))
    if inner_folds < 2:
        raise ValueError("At least two groups are required for group-aware validation")
    splitter = StratifiedGroupKFold(inner_folds, shuffle=True, random_state=seed)
    candidates = []
    positions = np.arange(len(development))
    for train_rel, val_rel in splitter.split(positions, y[development], group_values):
        if len(np.unique(y[development][train_rel])) != len(np.unique(y)):
            continue
        if len(np.unique(y[development][val_rel])) != len(np.unique(y)):
            continue
        candidates.append((train_rel, val_rel))
    if not candidates:
        raise ValueError("Unable to build a stratified group-aware validation split")
    train_rel, val_rel = min(candidates, key=lambda pair: abs(len(pair[1]) / len(development) - config.val_ratio))
    train_idx, val_idx = development[train_rel], development[val_rel]
    if np.intersect1d(groups[train_idx], groups[val_idx]).size:
        raise RuntimeError("Group leakage detected in validation split")
    return train_idx, val_idx


def fold_stem(repeat: int, fold: int, repeats: int) -> str:
    return f"fold_{fold}" if repeats == 1 else f"repeat_{repeat}_fold_{fold}"


def build_callbacks(
    config: TrainConfig,
    checkpoint: Path,
    x_val: np.ndarray,
    y_val_hard: np.ndarray,
) -> list[tf.keras.callbacks.Callback]:
    monitor = config.checkpoint_metric
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=config.early_stopping_patience,
            restore_best_weights=monitor not in {"val_macro_f1", "val_accuracy"},
            mode="max" if monitor in {"val_macro_f1", "val_accuracy"} else "auto",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=config.reduce_lr_patience,
            factor=0.5,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.CSVLogger(checkpoint.with_name(checkpoint.stem + "_history.csv")),
    ]
    if monitor == "val_macro_f1":
        callbacks.insert(
            0,
            MacroF1Checkpoint(
                checkpoint,
                x_val,
                y_val_hard,
                calibrate=False,
            ),
        )
    elif monitor == "val_accuracy":
        callbacks.insert(0, AccuracyCheckpoint(checkpoint, x_val, y_val_hard))
    else:
        callbacks.insert(
            0,
            tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor=monitor, save_best_only=True),
        )
    return callbacks


def run_cross_validation(config: TrainConfig) -> dict[str, Any]:
    logger = setup_logging()
    config.validate()
    set_seed(config.seed)
    out = Path(config.output)
    out.mkdir(parents=True, exist_ok=True)
    config.save(out / "config.json")

    x = np.load(config.data).astype(np.float32)
    labels = np.load(config.labels)
    if x.ndim != 3 or len(x) != len(labels) or not np.isfinite(x).all():
        raise ValueError("Expected finite X[N,C,T] and matching labels")

    groups = load_groups(config, labels)
    data_meta = {
        "data_checksum": data_checksum(config.data),
        "labels_checksum": data_checksum(config.labels),
        "samples": int(len(labels)),
        "class_distribution": {
            str(label): int(count)
            for label, count in zip(*np.unique(labels, return_counts=True))
        },
        "split_mode": config.split_mode,
        "groups_checksum": data_checksum(config.groups) if config.groups else None,
    }
    (out / "data_manifest.json").write_text(json.dumps(data_meta, indent=2), encoding="utf-8")
    log_event(logger, "training_start", output=str(out), samples=len(labels))

    classes, y = np.unique(labels, return_inverse=True)
    raw_channels = x.shape[1]
    raw_samples = x.shape[2]
    x = transform_model_input(x, config)
    if x.ndim == 3:
        x = x[..., None]
    fold_metrics: list[dict[str, Any]] = []
    oof_probability_sum = np.zeros((len(y), len(classes)), dtype=np.float64)
    oof_count = np.zeros(len(y), dtype=np.int32)
    oof_fold = np.full(len(y), -1, dtype=np.int32)

    for repeat, fold, development, test in iter_outer_splits(config, y, groups):
        train_idx, val_idx = split_train_validation(
            development,
            y,
            groups,
            config,
            config.seed + (repeat - 1) * config.folds + fold,
        )
        (x_train, x_val, x_test), mean, std = normalize_from_train(
            x[train_idx], x[val_idx], x[test]
        )
        y_train = tf.keras.utils.to_categorical(y[train_idx], len(classes))
        y_val = tf.keras.utils.to_categorical(y[val_idx], len(classes))
        model = build_model(config, len(classes), x.shape[1], x.shape[2])
        compile_model(model, config, y[train_idx])

        stem = fold_stem(repeat, fold, config.repeats)
        checkpoint = out / f"{stem}.keras"
        callbacks = build_callbacks(config, checkpoint, x_val, y[val_idx])

        model.fit(
            EEGSequence(
                x_train,
                y_train,
                config.batch_size,
                config.seed + fold,
                augment=True,
                mixup_alpha=config.mixup_alpha,
                oversample=config.oversample,
                minority_boost=config.minority_boost,
            ),
            validation_data=(x_val, y_val),
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=2,
        )

        if config.checkpoint_metric in {"val_macro_f1", "val_accuracy"} and checkpoint.exists():
            model = tf.keras.models.load_model(
                checkpoint,
                custom_objects={
                    "CategoricalFocalLoss": CategoricalFocalLoss,
                    "WeightedCategoricalCrossentropy": WeightedCategoricalCrossentropy,
                },
            )

        val_probability = model.predict(x_val, verbose=0)
        test_probability = model.predict(x_test, verbose=0)
        class_multipliers = np.ones(len(classes), dtype=np.float32)
        if config.calibrate:
            class_multipliers, _ = tune_class_multipliers(
                val_probability,
                y[val_idx],
                target=config.selection_metric,
            )

        predicted = test_probability.argmax(axis=1)
        metrics = {"repeat": repeat, "fold": fold, **evaluate_fold(y[test], predicted, classes)}
        if config.calibrate:
            calibrated = predict_with_multipliers(test_probability, class_multipliers)
            calibrated_metrics = evaluate_fold(y[test], calibrated, classes)
            metrics["calibrated_macro_f1"] = calibrated_metrics["macro_f1"]
            metrics["calibrated_accuracy"] = calibrated_metrics["accuracy"]
            metrics["calibrated_balanced_accuracy"] = calibrated_metrics["balanced_accuracy"]
            metrics["class_multipliers"] = class_multipliers.tolist()
            metrics["primary_macro_f1"] = metrics["calibrated_macro_f1"]
            metrics["primary_accuracy"] = metrics["calibrated_accuracy"]
        else:
            metrics["primary_macro_f1"] = metrics["macro_f1"]
            metrics["primary_accuracy"] = metrics["accuracy"]
        fold_metrics.append(metrics)

        oof_probability_sum[test] += test_probability
        oof_count[test] += 1
        oof_fold[test] = (repeat - 1) * config.folds + fold

        save_preprocessing_bundle(
            out / f"{stem}_preprocessing.npz",
            mean,
            std,
            classes,
            config,
            class_multipliers,
            raw_channels,
            raw_samples,
        )
        (out / f"{stem}_metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        log_event(
            logger,
            "fold_complete",
            repeat=repeat,
            fold=fold,
            macro_f1=metrics["primary_macro_f1"],
            accuracy=metrics["primary_accuracy"],
        )

    if not np.all(oof_count == config.repeats):
        raise RuntimeError("OOF predictions do not cover every sample for every repeat")
    oof_probability = (oof_probability_sum / oof_count[:, None]).astype(np.float32)
    np.savez_compressed(
        out / "oof_predictions.npz",
        sample_indices=np.arange(len(y), dtype=np.int64),
        labels=y.astype(np.int64),
        classes=classes,
        probabilities=oof_probability,
        prediction_counts=oof_count,
        fold_ids=oof_fold,
        groups=groups if groups is not None else np.array([], dtype=np.int64),
    )

    protocol = config.protocol_note
    if config.split_mode == "group":
        protocol = f"group-aware cross-validation ({Path(config.groups).name})"
    if config.repeats > 1:
        protocol = f"{protocol}; {config.repeats} repeats"
    summary = summarize_folds(fold_metrics, protocol)
    for prefix in ("calibrated_macro_f1", "calibrated_accuracy", "calibrated_balanced_accuracy"):
        values = [fold[prefix] for fold in fold_metrics if prefix in fold]
        if values:
            summary[prefix] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    if any("primary_macro_f1" in fold for fold in fold_metrics):
        summary["primary_macro_f1"] = {
            "mean": float(np.mean([fold["primary_macro_f1"] for fold in fold_metrics])),
            "std": float(np.std([fold["primary_macro_f1"] for fold in fold_metrics])),
        }
        summary["primary_accuracy"] = {
            "mean": float(np.mean([fold["primary_accuracy"] for fold in fold_metrics])),
            "std": float(np.std([fold["primary_accuracy"] for fold in fold_metrics])),
        }
    oof_metrics = evaluate_fold(y, oof_probability.argmax(axis=1), classes)
    summary["oof_raw"] = {
        key: oof_metrics[key]
        for key in ("accuracy", "balanced_accuracy", "macro_f1")
    }
    summary["oof_raw_confidence_intervals"] = bootstrap_confidence_intervals(
        y,
        oof_probability.argmax(axis=1),
        config.bootstrap_iterations,
        config.seed,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    from eeg_project.report import render_markdown_report

    render_markdown_report(out / "summary.json", out / "report.md")
    log_event(
        logger,
        "training_complete",
        macro_f1=summary.get("primary_macro_f1", summary.get("macro_f1")),
        accuracy=summary.get("primary_accuracy", summary.get("accuracy")),
    )
    return summary


def train_production_model(
    config: TrainConfig,
    output_path: str | Path,
    preprocessing_path: str | Path,
) -> tf.keras.Model:
    """Train a deployment model on all labeled data with a held-out validation split."""
    logger = setup_logging()
    config.validate()
    set_seed(config.seed)
    x = np.load(config.data).astype(np.float32)
    labels = np.load(config.labels)
    classes, y = np.unique(labels, return_inverse=True)
    raw_channels = x.shape[1]
    raw_samples = x.shape[2]
    x = transform_model_input(x, config)
    if x.ndim == 3:
        x = x[..., None]
    groups = load_groups(config, labels)
    train_idx, val_idx = split_train_validation(
        np.arange(len(y)), y, groups, config, config.seed
    )
    (x_train, x_val), mean, std = normalize_from_train(x[train_idx], x[val_idx])
    y_train = tf.keras.utils.to_categorical(y[train_idx], len(classes))
    y_val = tf.keras.utils.to_categorical(y[val_idx], len(classes))

    model = build_model(config, len(classes), x.shape[1], x.shape[2])
    compile_model(model, config, y[train_idx])

    output_path = Path(output_path)
    checkpoint = output_path.with_suffix(".checkpoint.keras")
    callbacks = build_callbacks(config, checkpoint, x_val, y[val_idx])
    model.fit(
        EEGSequence(
            x_train,
            y_train,
            config.batch_size,
            config.seed,
            augment=True,
            mixup_alpha=config.mixup_alpha,
            oversample=config.oversample,
            minority_boost=config.minority_boost,
        ),
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    if config.checkpoint_metric in {"val_macro_f1", "val_accuracy"} and checkpoint.exists():
        model = tf.keras.models.load_model(
            checkpoint,
            custom_objects={
                "CategoricalFocalLoss": CategoricalFocalLoss,
                "WeightedCategoricalCrossentropy": WeightedCategoricalCrossentropy,
            },
        )

    class_multipliers = np.ones(len(classes), dtype=np.float32)
    if config.calibrate:
        val_probability = model.predict(x_val, verbose=0)
        class_multipliers, _ = tune_class_multipliers(
            val_probability,
            y[val_idx],
            target=config.selection_metric,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    save_preprocessing_bundle(
        preprocessing_path,
        mean,
        std,
        classes,
        config,
        class_multipliers,
        raw_channels,
        raw_samples,
    )

    manifest = build_manifest(
        output_path,
        preprocessing_path,
        config=json.loads(config.to_json()),
    )
    save_manifest(manifest, output_path.parent / "manifest.json")
    log_event(logger, "production_model_saved", model=str(output_path))
    return model
