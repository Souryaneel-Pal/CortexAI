import numpy as np
import torch

from src.explain.shap_tab import (
    ClassLogitsOnly,
    combine_shap_and_attention,
    rank_features_by_mean_abs_shap,
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


# LightGBM and PyTorch each bundle their own OpenMP runtime, and on
# macOS/ARM with this project's pinned versions (torch 2.13, lightgbm 4.7,
# Python 3.14) having both mapped into one process hard-crashes the
# interpreter -- a SIGSEGV, not a catchable exception, which takes the whole
# pytest session down with it. Import order does not fix it (it only moves
# the crash from lightgbm into torch), and neither OMP_NUM_THREADS=1 nor
# KMP_DUPLICATE_LIB_OK=TRUE help.
#
# The LightGBM SHAP path is real production code (LightGBMStack in
# src/models/tabular_ft.py), so it is genuinely exercised here rather than
# skipped -- just in a subprocess that never imports torch.
_LIGHTGBM_SHAP_SUBPROCESS = """
# lightgbm MUST be imported before anything that pulls in torch --
# src.explain.shap_tab imports torch at module level, and with torch's libomp
# already mapped, LightGBM's first Dataset construction segfaults. Importing
# lightgbm first binds its own OpenMP runtime and both then coexist for the
# duration of this process.
import lightgbm as lgb
import numpy as np

from src.explain.shap_tab import shap_values_for_lightgbm
from src.data.schemas import TABULAR_FEATURE_COLUMNS

n_features = len(TABULAR_FEATURE_COLUMNS)
rng = np.random.default_rng(0)
X_train = rng.normal(size=(100, n_features))
y_train = rng.integers(0, 4, size=100)
clf = lgb.LGBMClassifier(n_estimators=10, verbosity=-1)
clf.fit(X_train, y_train)

X_test = rng.normal(size=(6, n_features))
values = shap_values_for_lightgbm(clf, X_test, class_index=0)
assert values.shape == (6, n_features), values.shape
assert np.isfinite(values).all()
print("OK")
"""


def test_shap_values_for_lightgbm_shape():
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", _LIGHTGBM_SHAP_SUBPROCESS],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"LightGBM SHAP subprocess failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout


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
