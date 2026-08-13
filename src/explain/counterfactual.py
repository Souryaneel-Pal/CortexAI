"""Counterfactual explanations over the 18 tabular features (docs/PROJECT_PLAN.md
P3, Objective 3 Level 3: "'If HRV rose by 12 and sleep quality reached 4,
predicted stress drops to Mild.' Turns attribution into guidance a clinician
can act on." -- docs/Proposal.pdf Sec. 06).

Primary: DiCE (Mothilal et al. 2020) via its PyTorch backend, wrapping the
tabular classifier directly (counterfactuals here are naturally defined on
the tabular features -- sleep quality, HRV, etc. -- the "actionable levers").

Fallback (documented, not silent -- used if dice-ml is unavailable or fails
to converge): a greedy single-feature grid search that perturbs one feature
at a time along its observed range and returns the smallest change that
flips the predicted class.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.schemas import (
    STRESS_LEVEL_NAMES,
    StressLevel,
    TABULAR_FEATURE_COLUMNS,
    TABULAR_TARGET_CLASS_COLUMN,
)


def _predict_class(model: torch.nn.Module, x: torch.Tensor) -> int:
    with torch.no_grad():
        _embedding, class_logits, _scores = model(x.unsqueeze(0))
        return int(class_logits.argmax(dim=-1).item())


def generate_counterfactual_dice(
    model: torch.nn.Module,
    background_df: pd.DataFrame,
    query_instance: dict[str, float],
    desired_class: int,
    total_cfs: int = 3,
    features_to_vary: list[str] | None = None,
) -> pd.DataFrame:
    """`background_df`: a DataFrame with the 18 feature columns + the
    Mental_Health_Status column (used by DiCE to learn feature ranges/types).
    `query_instance`: the participant's current feature values.
    `desired_class`: a target StressLevel index. DiCE's "opposite" shortcut
    only supports binary classification, so for our 4-class problem the
    caller always names the target tier explicitly.
    `features_to_vary`: restrict counterfactuals to actionable/behavioural
    features (e.g. Sleep_Quality) rather than physiological readings a
    person can't directly control -- pass a subset of TABULAR_FEATURE_COLUMNS.

    Uses DiCE's "random" (black-box query) search method rather than
    "gradient": the gradient method requires every layer to accept
    unbatched 1-D input during its internal optimization step, which fails
    against BatchNorm1d layers in our encoders (verified during development
    -- raises `ValueError: expected 2D or 3D input (got 1D input)`) -- the
    random method calls the model normally and has no such constraint.

    Returns a DataFrame of `total_cfs` counterfactual rows (feature values +
    predicted class), i.e. DiCE's native output shape.
    """
    import dice_ml

    class _SklearnLikeWrapper:
        """DiCE's PyTorch backend expects a `.model` attribute that is a
        plain nn.Module taking a single tensor and returning class scores.
        """

        def __init__(self, tabular_encoder: torch.nn.Module):
            self.tabular_encoder = tabular_encoder

        def __call__(self, x):
            _embedding, class_logits, _scores = self.tabular_encoder(x)
            return class_logits

    data = dice_ml.Data(
        dataframe=background_df,
        continuous_features=TABULAR_FEATURE_COLUMNS,
        outcome_name=TABULAR_TARGET_CLASS_COLUMN,
    )

    wrapped_model = _SklearnLikeWrapper(model)
    dice_model = dice_ml.Model(model=wrapped_model, backend="PYT")

    explainer = dice_ml.Dice(data, dice_model, method="random")
    query_df = pd.DataFrame([query_instance])[TABULAR_FEATURE_COLUMNS]

    result = explainer.generate_counterfactuals(
        query_df,
        total_CFs=total_cfs,
        desired_class=int(desired_class),
        features_to_vary=features_to_vary or "all",
    )
    return result.cf_examples_list[0].final_cfs_df


def generate_counterfactual_grid_search(
    model: torch.nn.Module,
    query_instance: dict[str, float],
    feature_ranges: dict[str, tuple[float, float]],
    desired_class: int,
    features_to_vary: list[str] | None = None,
    n_grid_points: int = 20,
) -> dict | None:
    """Fallback counterfactual search (used when dice-ml is unavailable/fails):
    perturbs one feature at a time across `feature_ranges[feature]`, greedily
    picking the single-feature, smallest-magnitude change that flips the
    prediction to `desired_class`. Only ever proposes one lever at a time --
    strictly simpler than DiCE's joint multi-feature search, documented as
    the degraded mode.

    Returns {"feature": str, "from": float, "to": float, "new_class": int}
    for the smallest successful single-feature change, or None if no single
    feature in `features_to_vary` flips the class within its observed range.
    """
    features_to_vary = features_to_vary or list(feature_ranges.keys())
    base_vector = [query_instance[f] for f in TABULAR_FEATURE_COLUMNS]
    base_class = _predict_class(model, torch.tensor(base_vector, dtype=torch.float32))
    if base_class == desired_class:
        return None

    best: dict | None = None
    for feature in features_to_vary:
        low, high = feature_ranges[feature]
        original_value = query_instance[feature]
        feature_idx = TABULAR_FEATURE_COLUMNS.index(feature)

        for candidate in np.linspace(low, high, n_grid_points):
            trial_vector = list(base_vector)
            trial_vector[feature_idx] = float(candidate)
            predicted = _predict_class(model, torch.tensor(trial_vector, dtype=torch.float32))
            if predicted == desired_class:
                magnitude = abs(candidate - original_value)
                if best is None or magnitude < best["_magnitude"]:
                    best = {
                        "feature": feature,
                        "from": float(original_value),
                        "to": float(candidate),
                        "new_class": desired_class,
                        "_magnitude": magnitude,
                    }
                break  # smallest change in this feature's direction found; move to next feature

    if best is not None:
        best.pop("_magnitude")
    return best


def format_counterfactual_narrative(counterfactual: dict) -> str:
    """Turns a grid-search counterfactual dict into the user-facing sentence
    from the docs: "If sleep quality rose from 2 to 4, predicted stress
    drops to Mild."
    """
    new_class_name = STRESS_LEVEL_NAMES[StressLevel(counterfactual["new_class"])].replace("_", " ")
    direction = "rose" if counterfactual["to"] > counterfactual["from"] else "dropped"
    feature_label = counterfactual["feature"].replace("_", " ")
    return (
        f"If {feature_label} {direction} from {counterfactual['from']:.1f} to "
        f"{counterfactual['to']:.1f}, predicted status moves to {new_class_name}."
    )
