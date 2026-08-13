"""Evaluation harness matching docs/Metrics_Used.docx exactly.

Classification: Accuracy, Precision, Recall/Sensitivity, F1-Score,
                 Macro F1-Score, Weighted F1-Score, ROC-AUC, Confusion Matrix.
Regression:      MAE, MSE, RMSE, R2 Score, Explained Variance Score
                 (reported per target: Depression, Anxiety, Stress).

Headline metrics (imbalanced classes / outlier-sensitive scores, per
docs/MINDSCOPE_Blueprint.pdf Sec. 11 and docs/Proposal.pdf Sec. 07):
Macro-F1 for classification, RMSE for regression.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.data.schemas import StressLevel, TABULAR_TARGET_SCORE_COLUMNS

N_CLASSES = len(StressLevel)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision_per_class: list[float]
    recall_per_class: list[float]
    f1_per_class: list[float]
    macro_f1: float
    weighted_f1: float
    roc_auc_macro: float | None
    confusion_matrix: list[list[int]]

    @property
    def headline(self) -> float:
        """Macro-F1 is the headline metric (imbalanced classes)."""
        return self.macro_f1


@dataclass
class RegressionMetrics:
    mae: float
    mse: float
    rmse: float
    r2: float
    explained_variance: float

    @property
    def headline(self) -> float:
        """RMSE is the headline metric (original score units, outlier-sensitive)."""
        return self.rmse


@dataclass
class MultiTargetRegressionMetrics:
    per_target: dict[str, RegressionMetrics] = field(default_factory=dict)

    @property
    def mean_rmse(self) -> float:
        return float(np.mean([m.rmse for m in self.per_target.values()]))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> ClassificationMetrics:
    """y_true, y_pred: 1-D int arrays of class indices in [0, N_CLASSES).
    y_proba: optional (n_samples, N_CLASSES) probability matrix for ROC-AUC.
    """
    labels = list(range(N_CLASSES))

    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = float(
                roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class="ovr",
                    average="macro",
                    labels=labels,
                )
            )
        except ValueError:
            # Fewer than N_CLASSES present in y_true (small/synthetic eval batch) --
            # ROC-AUC is undefined; leave as None rather than fabricate a number.
            roc_auc = None

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_per_class=precision_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ).tolist(),
        recall_per_class=recall_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ).tolist(),
        f1_per_class=f1_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ).tolist(),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        roc_auc_macro=roc_auc,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    )


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """y_true, y_pred: 1-D float arrays for a single target."""
    mse = float(mean_squared_error(y_true, y_pred))
    return RegressionMetrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        mse=mse,
        rmse=float(np.sqrt(mse)),
        r2=float(r2_score(y_true, y_pred)),
        explained_variance=float(explained_variance_score(y_true, y_pred)),
    )


def compute_multitarget_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str] | None = None
) -> MultiTargetRegressionMetrics:
    """y_true, y_pred: (n_samples, n_targets) arrays. Reports metrics per target,
    matching the "reported separately for Depression, Anxiety and Stress"
    requirement (docs/Proposal.pdf Sec. 07).
    """
    target_names = target_names or TABULAR_TARGET_SCORE_COLUMNS
    n_targets = y_true.shape[1]
    if len(target_names) != n_targets:
        raise ValueError(
            f"target_names has {len(target_names)} entries but arrays have {n_targets} targets"
        )
    result = MultiTargetRegressionMetrics()
    for i, name in enumerate(target_names):
        result.per_target[name] = compute_regression_metrics(y_true[:, i], y_pred[:, i])
    return result
