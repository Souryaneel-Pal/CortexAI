import numpy as np
import pandas as pd
import torch

from src.explain.counterfactual import (
    format_counterfactual_narrative,
    generate_counterfactual_dice,
    generate_counterfactual_grid_search,
)
from src.data.schemas import TABULAR_FEATURE_COLUMNS, TABULAR_TARGET_CLASS_COLUMN
from src.models.tabular_ft import TabularEncoder


class _ThresholdOnSleepQuality(torch.nn.Module):
    """A deterministic stand-in model: predicts class 3 (Severe) if
    Sleep_Quality < 3, else class 0 (Healthy) -- makes the grid search's
    correctness independently verifiable against a known decision boundary,
    rather than trusting an untrained real encoder's arbitrary boundary.
    """

    def __init__(self):
        super().__init__()
        self.sleep_quality_idx = TABULAR_FEATURE_COLUMNS.index("Sleep_Quality")

    def forward(self, x: torch.Tensor):
        sleep_quality = x[:, self.sleep_quality_idx]
        class_idx = torch.where(sleep_quality < 3, torch.tensor(3), torch.tensor(0))
        logits = torch.nn.functional.one_hot(class_idx, num_classes=4).float() * 10.0
        return None, logits, None


def test_grid_search_finds_sleep_quality_flip():
    model = _ThresholdOnSleepQuality()
    query = {f: 0.0 for f in TABULAR_FEATURE_COLUMNS}
    query["Sleep_Quality"] = 2.0  # below threshold -> currently predicted Severe (3)

    result = generate_counterfactual_grid_search(
        model,
        query_instance=query,
        feature_ranges={"Sleep_Quality": (1.0, 5.0)},
        desired_class=0,
        features_to_vary=["Sleep_Quality"],
        n_grid_points=50,
    )
    assert result is not None
    assert result["feature"] == "Sleep_Quality"
    assert result["to"] >= 3.0
    assert result["new_class"] == 0


def test_grid_search_returns_none_when_already_desired_class():
    model = _ThresholdOnSleepQuality()
    query = {f: 0.0 for f in TABULAR_FEATURE_COLUMNS}
    query["Sleep_Quality"] = 4.0  # already predicted Healthy (0)

    result = generate_counterfactual_grid_search(
        model,
        query_instance=query,
        feature_ranges={"Sleep_Quality": (1.0, 5.0)},
        desired_class=0,
        features_to_vary=["Sleep_Quality"],
    )
    assert result is None


def test_grid_search_returns_none_when_no_feature_can_flip_it():
    model = _ThresholdOnSleepQuality()
    query = {f: 0.0 for f in TABULAR_FEATURE_COLUMNS}
    query["Sleep_Quality"] = 2.0

    result = generate_counterfactual_grid_search(
        model,
        query_instance=query,
        feature_ranges={"HRV_Index": (0.0, 100.0)},  # doesn't affect this model's decision
        desired_class=0,
        features_to_vary=["HRV_Index"],
    )
    assert result is None


def test_format_counterfactual_narrative_matches_docs_style():
    counterfactual = {"feature": "Sleep_Quality", "from": 2.0, "to": 4.0, "new_class": 1}
    narrative = format_counterfactual_narrative(counterfactual)
    assert "Sleep Quality" in narrative
    assert "2.0" in narrative and "4.0" in narrative
    assert "Mild Stress" in narrative


def test_grid_search_against_real_untrained_encoder_runs_without_error():
    torch.manual_seed(0)
    model = TabularEncoder(backbone="residual_mlp")
    model.eval()
    query = {f: 0.0 for f in TABULAR_FEATURE_COLUMNS}
    ranges = {f: (-2.0, 2.0) for f in TABULAR_FEATURE_COLUMNS}
    # Just verify it runs end-to-end against a real model without crashing;
    # an untrained model's decision boundary is arbitrary, so we don't
    # assert on the specific outcome here.
    generate_counterfactual_grid_search(
        model, query_instance=query, feature_ranges=ranges, desired_class=2, n_grid_points=5
    )


def test_dice_random_method_produces_requested_number_of_counterfactuals():
    torch.manual_seed(0)
    model = TabularEncoder(backbone="residual_mlp")
    model.eval()

    rng = np.random.default_rng(0)
    n = 60
    classes = ["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"]
    data = {col: rng.normal(size=n) for col in TABULAR_FEATURE_COLUMNS}
    data[TABULAR_TARGET_CLASS_COLUMN] = rng.choice(classes, size=n)
    df = pd.DataFrame(data)

    query = {col: 0.0 for col in TABULAR_FEATURE_COLUMNS}
    result = generate_counterfactual_dice(model, df, query, desired_class=1, total_cfs=2)
    assert len(result) == 2
    assert (result[TABULAR_TARGET_CLASS_COLUMN] == 1).all()
