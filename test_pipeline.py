import numpy as np
import pytest
from fastapi.testclient import TestClient

from preprocess import filter_eeg, validate_data
from eeg_project.augment import EEGSequence
from eeg_project.config import TrainConfig
from eeg_project.metrics import evaluate_fold, summarize_folds
from eeg_project.training import iter_outer_splits, normalize_from_train, run_cross_validation


def test_filter_shape_and_finite():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(8, 5, 1000)).astype(np.float32)
    y = np.arange(8) % 3
    validate_data(x, y)
    result = filter_eeg(x, 250, 4, 40, 50)
    assert result.shape == x.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_invalid_shape_rejected():
    with pytest.raises(ValueError):
        validate_data(np.zeros((5, 1000)), np.zeros(5))


def test_normalize_uses_train_statistics_only():
    rng = np.random.default_rng(0)
    train = rng.normal(loc=2.0, scale=0.5, size=(4, 5, 10)).astype(np.float32)
    val = rng.normal(loc=4.0, scale=0.5, size=(2, 5, 10)).astype(np.float32)
    (norm_train, norm_val), _, _ = normalize_from_train(train, val)
    assert abs(float(norm_train.mean())) < 0.3
    assert float(norm_val.mean()) > 0.5


def test_eeg_sequence_mixup_preserves_batch_shape():
    x = np.random.randn(16, 5, 100, 1).astype(np.float32)
    y = np.eye(3)[np.random.randint(0, 3, 16)]
    sequence = EEGSequence(x, y, batch_size=8, seed=1, augment=True, mixup_alpha=0.2)
    batch_x, batch_y = sequence[0]
    assert batch_x.shape[0] <= 8
    assert batch_y.shape[1] == 3


def test_metrics_summary():
    folds = [
        {"accuracy": 0.6, "balanced_accuracy": 0.58, "macro_f1": 0.57},
        {"accuracy": 0.7, "balanced_accuracy": 0.68, "macro_f1": 0.67},
    ]
    summary = summarize_folds(folds, "test")
    assert summary["accuracy"]["mean"] == pytest.approx(0.65)
    assert summary["protocol"] == "test"


def test_train_config_yaml_roundtrip(tmp_path):
    config = TrainConfig.from_yaml("configs/eegnet.yaml")
    path = tmp_path / "config.json"
    config.save(path)
    assert "sample-level stratified" in path.read_text(encoding="utf-8")


def test_cross_validation_smoke(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(96, 5, 128)).astype(np.float32)
    y = rng.integers(0, 3, size=96)
    data = tmp_path / "x.npy"
    labels = tmp_path / "y.npy"
    np.save(data, x)
    np.save(labels, y)

    config = TrainConfig(
        data=str(data),
        labels=str(labels),
        output=str(tmp_path / "artifacts"),
        folds=2,
        epochs=1,
        batch_size=16,
        model=TrainConfig().model,
    )
    summary = run_cross_validation(config)
    assert "accuracy" in summary
    assert len(summary["folds"]) == 2
    with np.load(tmp_path / "artifacts" / "oof_predictions.npz", allow_pickle=False) as oof:
        assert oof["probabilities"].shape == (96, 3)
        assert (oof["prediction_counts"] == 1).all()


def test_group_cross_validation_has_no_subject_overlap(tmp_path):
    labels = np.repeat(np.arange(3), 10)
    groups = np.tile(np.arange(10), 3)
    groups_path = tmp_path / "groups.npy"
    np.save(groups_path, groups)
    config = TrainConfig(folds=5, split_mode="group", groups=str(groups_path))
    for _, _, development, test in iter_outer_splits(config, labels, groups):
        assert not np.intersect1d(groups[development], groups[test]).size


def test_spectral_eegnet_branch_builds():
    from arl_eegmodels.EEGModels import EEGNet

    model = EEGNet(
        3,
        Chans=5,
        Samples=128,
        spectral_features=True,
        spectral_bands=[(4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 40.0)],
    )
    output = model(np.zeros((2, 5, 128, 1), dtype=np.float32), training=False).numpy()
    assert output.shape == (2, 3)
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5)


def test_fbcsp_extractor_fits_on_training_data_only():
    from eeg_project.features import FBCSPFeatureExtractor

    rng = np.random.default_rng(42)
    x = rng.normal(size=(18, 5, 128)).astype(np.float32)
    y = np.repeat(np.arange(3), 6)
    extractor = FBCSPFeatureExtractor(fs=250.0)
    train_features = extractor.fit_transform(x[:15], y[:15])
    test_features = extractor.transform(x[15:])
    # Five bands x three one-vs-rest filters x four selected CSP components.
    assert train_features.shape == (15, 60)
    assert test_features.shape == (3, 60)
    assert np.isfinite(test_features).all()


def test_calibration_improves_minority_class():
    from eeg_project.calibration import predict_with_multipliers, tune_class_multipliers

    probability = np.array(
        [
            [0.30, 0.30, 0.40],
            [0.25, 0.25, 0.50],
            [0.20, 0.20, 0.60],
            [0.10, 0.10, 0.80],
        ],
        dtype=np.float32,
    )
    truth = np.array([2, 2, 2, 2])
    multipliers, score = tune_class_multipliers(probability, truth)
    predicted = predict_with_multipliers(probability, multipliers)
    assert score >= 0.75
    assert (predicted == 2).sum() >= 2


def test_ovr_falls_back_to_highest_probability_when_no_class_passes():
    from eeg_project.calibration import normalize_scores, ovr_predict

    probability = np.array([[0.20, 0.50, 0.30], [0.40, 0.25, 0.35]], dtype=np.float32)
    predicted = ovr_predict(probability, np.array([0.8, 0.8, 0.8], dtype=np.float32))
    assert predicted.tolist() == [1, 0]
    adjusted = normalize_scores(probability * np.array([[1.0, 2.0, 0.5]], dtype=np.float32))
    assert np.allclose(adjusted.sum(axis=1), 1.0)


def test_hybrid_policy_cross_fits_on_probability_rows():
    from eeg_project.hybrid import cross_fitted_hybrid_metrics, fit_hybrid_policy, fuse_probabilities

    labels = np.repeat(np.arange(3), 12)
    classes = np.arange(3)
    eegnet = np.full((len(labels), 3), 0.1, dtype=np.float32)
    fbcsp = np.full((len(labels), 3), 0.1, dtype=np.float32)
    eegnet[np.arange(len(labels)), labels] = 0.8
    fbcsp[np.arange(len(labels)), labels] = 0.7
    eegnet /= eegnet.sum(axis=1, keepdims=True)
    fbcsp /= fbcsp.sum(axis=1, keepdims=True)

    policy = fit_hybrid_policy(eegnet, fbcsp, labels, target="accuracy")
    fused = fuse_probabilities(eegnet, fbcsp, policy["eegnet_weight"])
    result = cross_fitted_hybrid_metrics(eegnet, fbcsp, labels, classes, folds=3)

    assert fused.shape == eegnet.shape
    assert 0.0 <= policy["eegnet_weight"] <= 1.0
    assert result["prediction"].shape == labels.shape
    assert result["metrics"]["accuracy"] == pytest.approx(1.0)


def test_focal_loss_forward():
    import tensorflow as tf
    from eeg_project.losses import CategoricalFocalLoss

    loss_fn = CategoricalFocalLoss(gamma=2.0, alpha=[1.0, 1.0, 2.0])
    y_true = tf.constant([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    y_pred = tf.constant([[0.8, 0.1, 0.1], [0.2, 0.2, 0.6]])
    value = loss_fn(y_true, y_pred)
    assert float(tf.reduce_mean(value)) > 0


def test_api_health_without_model():
    from api import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 503
    assert client.get("/live").json()["status"] == "alive"


def test_inference_applies_saved_filter(tmp_path):
    from eeg_project.config import FilterConfig
    from eeg_project.signal import apply_filter, filter_config_to_arrays
    from inference import EEGPredictor

    rng = np.random.default_rng(0)
    raw = rng.normal(size=(2, 5, 64)).astype(np.float32)
    filtered = apply_filter(raw, FilterConfig())
    mean = filtered.mean(axis=(0, 2), keepdims=True)[..., None]
    std = filtered.std(axis=(0, 2), keepdims=True).clip(1e-6)[..., None]

    import tensorflow as tf
    from arl_eegmodels.EEGModels import EEGNet

    model = EEGNet(3, 5, 64)
    model_path = tmp_path / "model.keras"
    prep_path = tmp_path / "prep.npz"
    model.save(model_path)
    np.savez(
        prep_path,
        mean=mean,
        std=std,
        classes=np.array([0, 1, 2]),
        class_multipliers=np.ones(3, dtype=np.float32),
        **filter_config_to_arrays(FilterConfig()),
    )

    predictor = EEGPredictor(str(model_path), str(prep_path))
    assert predictor.apply_filtering is True
    prepared = predictor._prepare(raw)
    filtered = apply_filter(raw, FilterConfig())
    expected = ((filtered[..., None] - mean) / std).astype(np.float32)
    assert prepared.shape == expected.shape
    assert np.allclose(prepared, expected, atol=1e-4)


def test_minority_boost_changes_augmentation():
    x = np.random.randn(24, 5, 100, 1).astype(np.float32)
    labels = np.array([0] * 20 + [2] * 4)
    y = np.eye(3)[labels]
    sequence = EEGSequence(x, y, batch_size=24, seed=3, augment=True, minority_boost=True)
    batch_x, _ = sequence[0]
    assert batch_x.shape == x.shape
