"""Loss functions for CortexAI training (docs/MINDSCOPE_Blueprint.pdf Sec. 04).

- ClassBalancedFocalLoss: handles the FER Disgust (436 vs 7215) and RAVDESS
  Neutral (96 vs 192) imbalance for the classification tasks.
- HuberLoss wrapper: robust to score outliers for the regression tasks.
- UncertaintyWeightedMultiTaskLoss: auto-balances classification vs.
  regression loss magnitudes via learned per-task log-variance (Kendall,
  Gal & Cipolla 2018), so neither task dominates -- used in P2 fusion
  training (src/train/train_fusion.py); included here so both P1 (single
  modality, single task) and P2 share the same loss module.
- consistency_loss: nudges predicted scores to agree with the predicted
  class (e.g. Severe should not co-occur with a near-zero stress score).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.schemas import SCORE_RANGES, StressLevel, TABULAR_TARGET_SCORE_COLUMNS


def class_balanced_weights(class_counts: dict[str, int] | list[int], beta: float = 0.999) -> torch.Tensor:
    """Class-Balanced loss weights (Cui et al. 2019): (1-beta) / (1 - beta^n_c)."""
    counts = np.array(list(class_counts.values()) if isinstance(class_counts, dict) else class_counts, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)


class ClassBalancedFocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017) with class-balanced per-class weights."""

    def __init__(self, class_weights: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("class_weights", class_weights if class_weights is not None else None, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: (B, C), targets: (B,) int64 class indices."""
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_term = (1 - target_probs) ** self.gamma
        loss = -focal_term * target_log_probs

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)[targets]
            loss = loss * weights

        return loss.mean()


def score_regression_loss(preds: torch.Tensor, targets: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss over the 3 continuous score targets, robust to outliers."""
    return F.huber_loss(preds, targets, delta=delta)


class UncertaintyWeightedMultiTaskLoss(nn.Module):
    """Learns a log-variance per task and auto-balances the joint loss
    (Kendall, Gal & Cipolla 2018): L = sum_i [ exp(-log_var_i) * L_i + log_var_i ].
    Used for classification-loss + regression-loss balancing in fusion
    training so neither task dominates by raw magnitude.
    """

    def __init__(self, num_tasks: int = 2):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, task_losses: list[torch.Tensor]) -> torch.Tensor:
        assert len(task_losses) == len(self.log_vars)
        total = torch.zeros((), device=task_losses[0].device)
        for loss, log_var in zip(task_losses, self.log_vars):
            precision = torch.exp(-log_var)
            total = total + precision * loss + log_var
        return total


def consistency_loss(
    predicted_class_probs: torch.Tensor,
    predicted_scores: torch.Tensor,
    score_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalizes disagreement between the predicted class distribution and the
    predicted scores: computes the stress-severity tier each predicted score
    implies (as a soft, differentiable band membership) and compares it to
    the predicted class distribution via KL divergence.

    `predicted_class_probs`: (B, 4) softmax over StressLevel.
    `predicted_scores`: (B, 3) in the raw Depression/Anxiety/Stress units.
    Uses Stress_Score (the target on the same 0..max scale as the severity
    tiers) as the anchor, quartile-banded onto the 4-tier axis.
    """
    score_names = score_names or TABULAR_TARGET_SCORE_COLUMNS
    stress_idx = score_names.index("Stress_Score")
    stress_max = SCORE_RANGES["Stress_Score"][1]

    stress_norm = (predicted_scores[:, stress_idx] / stress_max).clamp(0, 1)

    # Soft band membership via 3 sigmoid boundaries at 0.25/0.5/0.75 of the
    # score range (Healthy|Mild|Moderate|Severe), differentiable everywhere.
    boundaries = torch.tensor([0.25, 0.5, 0.75], device=predicted_scores.device)
    sharpness = 25.0
    cdf = torch.sigmoid(sharpness * (stress_norm.unsqueeze(1) - boundaries))  # (B, 3)
    p_severe = cdf[:, 2]
    p_moderate = cdf[:, 1] - cdf[:, 2]
    p_mild = cdf[:, 0] - cdf[:, 1]
    p_healthy = 1 - cdf[:, 0]
    implied_class_dist = torch.stack(
        [p_healthy, p_mild, p_moderate, p_severe], dim=1
    ).clamp(min=1e-6)
    implied_class_dist = implied_class_dist / implied_class_dist.sum(dim=1, keepdim=True)

    predicted_class_probs = predicted_class_probs.clamp(min=1e-6)
    return F.kl_div(predicted_class_probs.log(), implied_class_dist, reduction="batchmean")
