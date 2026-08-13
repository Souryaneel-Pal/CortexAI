"""SHAP explanations for the tabular encoder (docs/PROJECT_PLAN.md P3,
Objective 3 Level 2: "SHAP ranks the 18 tabular signals" -- combined with
the FT-Transformer's own per-feature attention (src/models/tabular_ft.py's
`feature_attention_weights`) per docs/Proposal.pdf Sec. 06: "SHAP +
FT-Transformer attention -- Ranked feature bars".

Two backends, dispatched on what kind of model is passed in:
  - LightGBMStack (src/models/tabular_ft.py): shap.TreeExplainer -- fast,
    exact for tree ensembles.
  - TabularEncoder / any torch nn.Module: shap.GradientExplainer -- uses
    backprop, well suited to a differentiable model without the cost of a
    fully model-agnostic KernelExplainer.
"""
from __future__ import annotations

import numpy as np
import torch

from src.data.schemas import TABULAR_FEATURE_COLUMNS


def shap_values_for_lightgbm(classifier, X: np.ndarray, class_index: int) -> np.ndarray:
    """classifier: a fitted LightGBM classifier (e.g. LightGBMStack.classifier).
    Returns (n_samples, 18) SHAP values for the given class.
    """
    import shap

    explainer = shap.TreeExplainer(classifier)
    raw = explainer.shap_values(X)
    # Newer shap returns a (n_samples, n_features, n_classes) array for
    # multiclass; older versions return a list of per-class arrays.
    if isinstance(raw, list):
        return np.asarray(raw[class_index])
    if raw.ndim == 3:
        return raw[:, :, class_index]
    return raw


def shap_values_for_torch_model(
    model: torch.nn.Module,
    background: torch.Tensor,
    X: torch.Tensor,
    class_index: int,
) -> np.ndarray:
    """model: any torch module mapping (B, 18) features -> (B, num_classes)
    logits directly (wrap TabularEncoder if it returns a tuple -- see
    `ClassLogitsOnly` below). `background`: a (n_background, 18) reference
    sample (configs/tabular.yaml's `shap_background_samples`).
    Returns (n_samples, 18) SHAP values for the given class.
    """
    import shap

    explainer = shap.GradientExplainer(model, background)
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        return np.asarray(raw[class_index])
    if raw.ndim == 3:
        return raw[:, :, class_index]
    return raw


class ClassLogitsOnly(torch.nn.Module):
    """Wraps TabularEncoder (which returns (embedding, class_logits, score_preds))
    down to just class_logits, since SHAP explainers expect a single-output
    callable.
    """

    def __init__(self, tabular_encoder: torch.nn.Module):
        super().__init__()
        self.tabular_encoder = tabular_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _embedding, class_logits, _scores = self.tabular_encoder(x)
        return class_logits


def rank_features_by_mean_abs_shap(shap_values: np.ndarray, feature_names: list[str] | None = None) -> list[tuple[str, float]]:
    """shap_values: (n_samples, 18). Returns [(feature_name, mean_abs_shap), ...]
    sorted descending -- the "ranked feature bars" shown to the user.
    """
    feature_names = feature_names or TABULAR_FEATURE_COLUMNS
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_abs.tolist()), key=lambda item: item[1], reverse=True)
    return ranked


def combine_shap_and_attention(
    shap_values: np.ndarray,
    attention_weights: torch.Tensor,
    feature_names: list[str] | None = None,
) -> list[dict]:
    """Merges SHAP's mean |value| ranking with the FT-Transformer's own
    per-feature attention (averaged over the batch) into one ranked list --
    two independent signals for the same 18 features, shown side by side so
    disagreement between them is visible rather than silently averaged away.
    """
    feature_names = feature_names or TABULAR_FEATURE_COLUMNS
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_attention = attention_weights.detach().cpu().numpy().mean(axis=0)

    combined = [
        {
            "feature": name,
            "mean_abs_shap": float(mean_abs_shap[i]),
            "mean_attention": float(mean_attention[i]),
        }
        for i, name in enumerate(feature_names)
    ]
    combined.sort(key=lambda item: item["mean_abs_shap"], reverse=True)
    return combined
