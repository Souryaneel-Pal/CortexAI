import numpy as np

from src.data.augment import class_balanced_sample_weights, oversample_minority_indices, spec_augment, tabular_smote


def test_oversample_minority_indices_balances_counts():
    labels = np.array([0] * 20 + [1] * 5)
    idx = oversample_minority_indices(labels)
    resampled_labels = labels[idx]
    counts = np.bincount(resampled_labels)
    assert counts[0] == counts[1] == 20


def test_class_balanced_sample_weights_favor_minority():
    labels = np.array([0] * 100 + [1] * 5)
    weights = class_balanced_sample_weights(labels)
    minority_weight = weights[labels == 1][0]
    majority_weight = weights[labels == 0][0]
    assert minority_weight > majority_weight


def test_tabular_smote_balances_classes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 18))
    y = np.array([0] * 100 + [1] * 20)
    X_res, y_res = tabular_smote(X, y)
    counts = np.bincount(y_res)
    assert counts[0] == counts[1]


def test_spec_augment_preserves_shape():
    spec = np.random.rand(64, 100).astype(np.float32)
    augmented = spec_augment(spec)
    assert augmented.shape == spec.shape
