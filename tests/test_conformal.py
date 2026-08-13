import numpy as np

from src.explain.conformal import (
    UncertaintyGate,
    calibrate_split_conformal,
    conformal_prediction_sets,
)


def _make_synthetic_probs(n, num_classes, rng, sharpness=8.0):
    """Generates class-probability rows peaked at a random true class, with
    controllable confidence (sharpness), plus the matching true labels.
    """
    labels = rng.integers(0, num_classes, size=n)
    logits = rng.normal(scale=0.5, size=(n, num_classes))
    logits[np.arange(n), labels] += sharpness
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs, labels


def test_split_conformal_achieves_target_coverage_empirically():
    rng = np.random.default_rng(0)
    cal_probs, cal_labels = _make_synthetic_probs(2000, num_classes=4, rng=rng, sharpness=2.0)
    calibration = calibrate_split_conformal(cal_probs, cal_labels, alpha=0.1)

    test_probs, test_labels = _make_synthetic_probs(5000, num_classes=4, rng=rng, sharpness=2.0)
    sets = conformal_prediction_sets(test_probs, calibration)

    covered = sum(1 for label, s in zip(test_labels, sets) if label in s)
    coverage = covered / len(test_labels)
    # Conformal prediction guarantees coverage >= 1 - alpha = 0.9 (marginally);
    # it does NOT guarantee tightness, so only the lower bound is meaningful.
    assert coverage >= 0.85


def test_conformal_sets_are_never_empty():
    rng = np.random.default_rng(1)
    cal_probs, cal_labels = _make_synthetic_probs(500, num_classes=4, rng=rng, sharpness=3.0)
    calibration = calibrate_split_conformal(cal_probs, cal_labels, alpha=0.1)

    test_probs, _ = _make_synthetic_probs(200, num_classes=4, rng=rng, sharpness=3.0)
    sets = conformal_prediction_sets(test_probs, calibration)
    assert all(len(s) >= 1 for s in sets)


def test_confident_predictions_yield_smaller_sets_than_uncertain_ones():
    rng = np.random.default_rng(2)
    cal_probs, cal_labels = _make_synthetic_probs(1000, num_classes=4, rng=rng, sharpness=3.0)
    calibration = calibrate_split_conformal(cal_probs, cal_labels, alpha=0.1)

    confident_probs, _ = _make_synthetic_probs(50, num_classes=4, rng=rng, sharpness=10.0)
    uncertain_probs, _ = _make_synthetic_probs(50, num_classes=4, rng=rng, sharpness=0.1)

    confident_sets = conformal_prediction_sets(confident_probs, calibration)
    uncertain_sets = conformal_prediction_sets(uncertain_probs, calibration)

    mean_confident_size = np.mean([len(s) for s in confident_sets])
    mean_uncertain_size = np.mean([len(s) for s in uncertain_sets])
    assert mean_confident_size < mean_uncertain_size


def test_uncertainty_gate_defers_on_low_confidence():
    from src.explain.conformal import ConformalCalibration

    calibration = ConformalCalibration(quantile=0.5, alpha=0.1)
    gate = UncertaintyGate(calibration, confidence_threshold=0.6)

    result = gate.should_defer(mc_dropout_confidence=0.4, prediction_set=[0])
    assert result["defer"] is True
    assert result["reason"] == "low_confidence"


def test_uncertainty_gate_defers_on_ambiguous_set():
    from src.explain.conformal import ConformalCalibration

    calibration = ConformalCalibration(quantile=0.5, alpha=0.1)
    gate = UncertaintyGate(calibration, confidence_threshold=0.6)

    result = gate.should_defer(mc_dropout_confidence=0.9, prediction_set=[0, 1])
    assert result["defer"] is True
    assert result["reason"] == "ambiguous_prediction_set"


def test_uncertainty_gate_does_not_defer_when_confident_and_unambiguous():
    from src.explain.conformal import ConformalCalibration

    calibration = ConformalCalibration(quantile=0.5, alpha=0.1)
    gate = UncertaintyGate(calibration, confidence_threshold=0.6)

    result = gate.should_defer(mc_dropout_confidence=0.9, prediction_set=[2])
    assert result["defer"] is False
    assert result["reason"] is None


def test_mapie_backend_produces_prediction_sets():
    rng = np.random.default_rng(3)
    cal_probs, cal_labels = _make_synthetic_probs(300, num_classes=4, rng=rng, sharpness=2.0)

    from src.explain.conformal import calibrate_mapie

    mapie_classifier = calibrate_mapie(cal_probs, cal_labels, alpha=0.1)
    test_indices = np.arange(20).reshape(-1, 1)
    _point_preds, pred_sets = mapie_classifier.predict_set(test_indices)
    # (n_samples, n_classes, n_confidence_levels) boolean membership array.
    assert pred_sets.shape[0] == 20
    assert pred_sets.shape[1] == 4
    assert pred_sets.dtype == bool
    # Every sample's set must be non-empty at the calibrated confidence level.
    assert (pred_sets[:, :, 0].sum(axis=1) >= 1).all()
