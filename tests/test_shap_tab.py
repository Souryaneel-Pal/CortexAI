import numpy as np
import torch

from src.explain.shap_tab import (
    ClassLogitsOnly,
    combine_shap_and_attention,
    rank_features_by_mean_abs_shap,
    shap_values_for_lightgbm,
    shap_values_for_torch_model,
)
from src.models.tabular_ft import NUM_FEATURES, TabularEncoder


def test_shap_values_for_torch_model_shape():
    torch.manual_seed(0)
    encoder = TabularEncoder(backbone="residual_mlp")
    wrapped = ClassLogitsOnly(encoder)
    wrapped.eval()

    background = torch.randn(20, NUM_FEATURES)
    X = torch.randn(5, NUM_FEATURES)
    values = shap_values_for_torch_model(wrapped, background, X, class_index=0)
    assert values.shape == (5, NUM_FEATURES)
    assert np.isfinite(values).all()


def test_shap_values_for_lightgbm_shape():
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(100, NUM_FEATURES))
    y_train = rng.integers(0, 4, size=100)
    clf = lgb.LGBMClassifier(n_estimators=10, verbosity=-1)
    clf.fit(X_train, y_train)

    X_test = rng.normal(size=(6, NUM_FEATURES))
    values = shap_values_for_lightgbm(clf, X_test, class_index=0)
    assert values.shape == (6, NUM_FEATURES)


def test_rank_features_by_mean_abs_shap_sorted_descending():
    rng = np.random.default_rng(0)
    shap_values = rng.normal(size=(30, NUM_FEATURES))
    ranked = rank_features_by_mean_abs_shap(shap_values)
    assert len(ranked) == NUM_FEATURES
    values = [v for _name, v in ranked]
    assert values == sorted(values, reverse=True)


def test_combine_shap_and_attention_merges_both_signals():
    rng = np.random.default_rng(0)
    shap_values = rng.normal(size=(10, NUM_FEATURES))
    attention = torch.rand(10, NUM_FEATURES)
    combined = combine_shap_and_attention(shap_values, attention)
    assert len(combined) == NUM_FEATURES
    assert set(combined[0].keys()) == {"feature", "mean_abs_shap", "mean_attention"}
    shap_scores = [item["mean_abs_shap"] for item in combined]
    assert shap_scores == sorted(shap_scores, reverse=True)
