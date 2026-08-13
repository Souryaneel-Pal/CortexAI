import numpy as np

from src.eval.metrics import (
    compute_classification_metrics,
    compute_multitarget_regression_metrics,
    compute_regression_metrics,
)


def test_classification_metrics_perfect_predictions():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = y_true.copy()
    m = compute_classification_metrics(y_true, y_pred)
    assert m.accuracy == 1.0
    assert m.macro_f1 == 1.0
    assert m.weighted_f1 == 1.0
    assert m.headline == 1.0
    assert sum(m.confusion_matrix[i][i] for i in range(4)) == 8


def test_classification_metrics_known_case():
    # Hand-computable: 4 samples, 1 class fully wrong.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 0])
    m = compute_classification_metrics(y_true, y_pred)
    assert m.accuracy == 0.75
    # class 0: precision = 2/3 (2 correct out of 3 predicted as 0), recall = 2/2 = 1.0
    assert round(m.precision_per_class[0], 4) == round(2 / 3, 4)
    assert m.recall_per_class[0] == 1.0
    # class 1: precision = 1/1 = 1.0, recall = 1/2 = 0.5
    assert m.precision_per_class[1] == 1.0
    assert m.recall_per_class[1] == 0.5


def test_regression_metrics_known_case():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 37.0])
    m = compute_regression_metrics(y_true, y_pred)
    expected_mae = np.mean(np.abs(y_true - y_pred))
    expected_mse = np.mean((y_true - y_pred) ** 2)
    assert round(m.mae, 6) == round(float(expected_mae), 6)
    assert round(m.mse, 6) == round(float(expected_mse), 6)
    assert round(m.rmse, 6) == round(float(np.sqrt(expected_mse)), 6)
    assert m.headline == m.rmse


def test_regression_metrics_perfect_predictions():
    y_true = np.array([1.0, 2.0, 3.0])
    m = compute_regression_metrics(y_true, y_true.copy())
    assert m.mae == 0.0
    assert m.rmse == 0.0
    assert m.r2 == 1.0
    assert m.explained_variance == 1.0


def test_multitarget_regression_reports_per_target():
    y_true = np.array([[10.0, 5.0, 20.0], [12.0, 6.0, 22.0]])
    y_pred = np.array([[11.0, 5.0, 19.0], [12.0, 7.0, 22.0]])
    result = compute_multitarget_regression_metrics(
        y_true, y_pred, target_names=["Depression_Score", "Anxiety_Score", "Stress_Score"]
    )
    assert set(result.per_target.keys()) == {"Depression_Score", "Anxiety_Score", "Stress_Score"}
    assert result.per_target["Anxiety_Score"].mae == 0.5
    assert result.mean_rmse > 0
