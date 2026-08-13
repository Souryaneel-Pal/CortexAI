import numpy as np
import pytest

from src.eval.fairness_audit import fairness_audit_by_gender


def test_fairness_audit_detects_no_gap_when_performance_equal():
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 4, size=n)
    y_pred = y_true.copy()  # perfect predictions for both groups
    genders = np.array(["male", "female"])[rng.integers(0, 2, size=n)]

    result = fairness_audit_by_gender(y_true, y_pred, genders.tolist())
    assert result.macro_f1_gap == pytest.approx(0.0, abs=1e-6)
    assert result.accuracy_gap == pytest.approx(0.0, abs=1e-6)
    assert set(result.per_group.keys()) == {"male", "female"}
    assert sum(result.group_sizes.values()) == n


def test_fairness_audit_detects_real_gap():
    rng = np.random.default_rng(0)
    n_per_group = 100
    male_true = rng.integers(0, 4, size=n_per_group)
    male_pred = male_true.copy()  # perfect for males

    female_true = rng.integers(0, 4, size=n_per_group)
    female_pred = rng.integers(0, 4, size=n_per_group)  # random for females

    y_true = np.concatenate([male_true, female_true])
    y_pred = np.concatenate([male_pred, female_pred])
    genders = ["male"] * n_per_group + ["female"] * n_per_group

    result = fairness_audit_by_gender(y_true, y_pred, genders)
    assert result.per_group["male"].macro_f1 > result.per_group["female"].macro_f1
    assert result.macro_f1_gap > 0.3
    assert "gap" in result.summary()


def test_fairness_audit_raises_with_single_group():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="at least 2 gender groups"):
        fairness_audit_by_gender(y_true, y_pred, ["male", "male", "male"])
