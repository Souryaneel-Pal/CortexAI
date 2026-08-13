"""Conformal uncertainty gate (docs/PROJECT_PLAN.md P3; docs/MINDSCOPE_Blueprint.pdf
Sec. 07 differentiator #2: "Statistically-valid prediction sets. When
confidence is low the system defers to a human instead of guessing -- the
single most important safety feature in clinical AI.")

Two conformal-method backends, selected via configs/fusion.yaml's
`conformal.method`:
  - "split_conformal" (default, dependency-free): a from-scratch split
    conformal classifier (Sadinle et al. 2019 style prediction sets) --
    calibrated once on held-out data, deterministic, easy to unit-test.
  - "mapie": wraps MAPIE's `MapieClassifier` in prefit mode over the same
    calibration split, for teams that want MAPIE's broader method selection
    (docs/MINDSCOPE_Blueprint.pdf Sec. 08 names MAPIE explicitly). Documented
    as an alternate path, not required for the gate to function.

Either backend produces a prediction SET (possibly >1 class) at a target
coverage level; `UncertaintyGate.should_defer` turns that + the MC-dropout
confidence (src/models/heads.py's `predict_with_uncertainty`) into the
human-review decision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalCalibration:
    """Holds the calibrated conformal quantile from a split-conformal fit."""

    quantile: float
    alpha: float


def calibrate_split_conformal(calibration_probs: np.ndarray, calibration_labels: np.ndarray, alpha: float = 0.1) -> ConformalCalibration:
    """Split conformal calibration via Adaptive Prediction Sets (APS, Romano
    et al. 2020): sort each calibration example's class probabilities
    descending and take the cumulative probability mass up to and including
    the true class. The (1-alpha) empirical quantile of these cumulative
    scores (with the standard finite-sample correction) becomes the mass
    threshold used to build prediction sets at inference.

    APS is used instead of plain LAC (threshold each class's raw
    probability independently) because LAC can degenerate: a near-uniform
    (maximally uncertain) prediction has every class probability below a
    confident threshold, which -- without a fallback -- yields an empty
    set, and any reasonable fallback (e.g. "keep the top class") then makes
    uncertain and confident predictions look identically sized. APS's
    cumulative-mass construction naturally grows the set for uncertain
    predictions and shrinks it for confident ones, with no fallback needed.

    `calibration_probs`: (n_cal, num_classes) softmax probabilities on a
    held-out calibration split (never the training or test split).
    `calibration_labels`: (n_cal,) true class indices for that split.
    `alpha`: miscoverage rate -- alpha=0.1 targets 90% coverage.

    Note on `calibrate_mapie`'s equivalent: MAPIE's `SplitConformalClassifier
    .predict_set(X)` returns `(point_predictions, prediction_sets)`, where
    `prediction_sets` is a boolean array of shape
    (n_samples, n_classes, n_confidence_levels) -- not a list of index
    lists like this function's `conformal_prediction_sets` -- convert with
    `np.where(prediction_sets[i, :, 0])[0].tolist()` per sample if switching
    backends.
    """
    n = len(calibration_labels)
    sort_order = np.argsort(-calibration_probs, axis=1)
    sorted_probs = np.take_along_axis(calibration_probs, sort_order, axis=1)
    cumulative = np.cumsum(sorted_probs, axis=1)

    true_class_rank = np.array(
        [np.where(sort_order[i] == calibration_labels[i])[0][0] for i in range(n)]
    )
    nonconformity = cumulative[np.arange(n), true_class_rank]

    # Finite-sample-corrected quantile level (Romano et al. 2019).
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    quantile = float(np.quantile(nonconformity, q_level, method="higher"))
    return ConformalCalibration(quantile=quantile, alpha=alpha)


def conformal_prediction_sets(probs: np.ndarray, calibration: ConformalCalibration) -> list[list[int]]:
    """probs: (n_samples, num_classes). Returns, per sample, the smallest
    prefix of classes (sorted by descending probability) whose cumulative
    probability mass reaches the calibrated quantile -- always non-empty by
    construction (the top class alone is included as soon as the cumulative
    sum starts). At the target coverage level, the true class is in this set
    with probability >= 1-alpha (marginally).
    """
    sets = []
    for row in probs:
        order = np.argsort(-row)
        cumulative = np.cumsum(row[order])
        cutoff = int(np.searchsorted(cumulative, calibration.quantile) + 1)
        cutoff = min(cutoff, len(row))
        sets.append(order[:cutoff].tolist())
    return sets


def calibrate_mapie(calibration_probs: np.ndarray, calibration_labels: np.ndarray, alpha: float = 0.1):
    """Alternate backend using MAPIE's prefit `SplitConformalClassifier`
    (MAPIE >= 1.0 API -- this project pins `mapie>=1.0.0`, see
    requirements.txt; the pre-1.0 `MapieClassifier` class this module used
    during earlier development no longer exists). Returns a calibrated
    `SplitConformalClassifier`; call `.predict_set(X)` to get prediction
    sets. Kept separate from `calibrate_split_conformal` so a MAPIE
    version/API change can't break the always-available default APS path.
    """
    from sklearn.base import BaseEstimator, ClassifierMixin
    from mapie.classification import SplitConformalClassifier

    class _PrefitProbsClassifier(BaseEstimator, ClassifierMixin):
        """Wraps precomputed probabilities as a fitted sklearn-style
        classifier, since MAPIE's prefit mode calls `.predict_proba(X)`
        where X is just an index into the already-computed probability
        array (our model itself is a torch module, not an sklearn estimator).
        """

        def __init__(self, probs: np.ndarray):
            self.probs = probs
            self.classes_ = np.arange(probs.shape[1])

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            indices = np.asarray(X).reshape(-1).astype(int)
            return self.probs[indices]

        def predict(self, X):
            return self.predict_proba(X).argmax(axis=1)

    estimator = _PrefitProbsClassifier(calibration_probs).fit(None, calibration_labels)
    mapie_classifier = SplitConformalClassifier(
        estimator=estimator, confidence_level=1 - alpha, prefit=True
    )
    calibration_indices = np.arange(len(calibration_labels)).reshape(-1, 1)
    mapie_classifier.conformalize(calibration_indices, calibration_labels)
    return mapie_classifier


class UncertaintyGate:
    """Combines MC-dropout confidence (src/models/heads.py) with a conformal
    prediction-set width to decide whether a prediction should defer to a
    human reviewer (docs/MINDSCOPE_Blueprint.pdf: "defer_to_human_below_confidence").
    """

    def __init__(self, calibration: ConformalCalibration, confidence_threshold: float = 0.6):
        self.calibration = calibration
        self.confidence_threshold = confidence_threshold

    def should_defer(self, mc_dropout_confidence: float, prediction_set: list[int]) -> dict:
        """Defers if EITHER the MC-dropout confidence is below threshold OR
        the conformal set contains more than one class (i.e. the model
        itself can't distinguish between >=2 stress tiers at the target
        coverage level) -- either signal alone is grounds for human review.
        """
        low_confidence = mc_dropout_confidence < self.confidence_threshold
        ambiguous_set = len(prediction_set) > 1
        return {
            "defer": bool(low_confidence or ambiguous_set),
            "reason": (
                "low_confidence" if low_confidence and not ambiguous_set
                else "ambiguous_prediction_set" if ambiguous_set and not low_confidence
                else "low_confidence_and_ambiguous_set" if low_confidence and ambiguous_set
                else None
            ),
            "mc_dropout_confidence": mc_dropout_confidence,
            "prediction_set_size": len(prediction_set),
        }
