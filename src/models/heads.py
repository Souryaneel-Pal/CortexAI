"""Multi-task heads shared by the fused representation (docs/MINDSCOPE_Blueprint.pdf
Sec. 04): a 4-class softmax classifier with MC-dropout uncertainty, and a
3-output regression head for Depression/Anxiety/Stress. Both consume the
256-d fused embedding from src/models/fusion.py (or, for the P1 tabular
baseline, the tabular encoder's own embedding directly).

The score<->class consistency term itself lives in src/train/losses.py
(`consistency_loss`) since it's a loss function, not a layer; this module
only produces the two predictions it compares.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.data.schemas import SCORE_RANGES, StressLevel, TABULAR_TARGET_SCORE_COLUMNS

NUM_CLASSES = len(StressLevel)
NUM_SCORE_TARGETS = len(TABULAR_TARGET_SCORE_COLUMNS)


class ClassificationHead(nn.Module):
    """4-class softmax head with MC-dropout for uncertainty estimation
    (docs/MINDSCOPE_Blueprint.pdf: "conformal + MC-dropout"). The dropout
    layer is always present in the module graph so `predict_with_uncertainty`
    can force it active at inference time regardless of `model.eval()`.
    """

    def __init__(self, embed_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(x))

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_passes: int = 20) -> dict[str, torch.Tensor]:
        """Runs `n_passes` stochastic forward passes with dropout forced on,
        returning the mean class probabilities and their epistemic
        (predictive) variance -- the input to the conformal uncertainty gate
        (src/explain/conformal.py) that decides whether to defer to a human.
        """
        was_training = self.dropout.training
        self.dropout.train()  # force dropout active even if the model is in eval()

        all_probs = torch.stack(
            [torch.softmax(self.forward(x), dim=-1) for _ in range(n_passes)], dim=0
        )  # (n_passes, B, NUM_CLASSES)

        self.dropout.train(was_training)

        mean_probs = all_probs.mean(dim=0)
        variance = all_probs.var(dim=0)
        predictive_entropy = -(mean_probs.clamp(min=1e-8) * mean_probs.clamp(min=1e-8).log()).sum(dim=-1)
        confidence = mean_probs.max(dim=-1).values
        return {
            "mean_probs": mean_probs,
            "variance": variance,
            "predictive_entropy": predictive_entropy,
            "confidence": confidence,
        }


class RegressionHead(nn.Module):
    """3-output regression head for Depression (0-34) / Anxiety (0-24) /
    Stress (0-39). Outputs are passed through a sigmoid and scaled to each
    target's documented range (docs/1.docx) so predictions are always
    physically valid, rather than relying on the loss alone to shape an
    unbounded linear output.
    """

    def __init__(self, embed_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_dim, NUM_SCORE_TARGETS)
        max_values = [SCORE_RANGES[name][1] for name in TABULAR_TARGET_SCORE_COLUMNS]
        self.register_buffer("max_values", torch.tensor(max_values, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.linear(self.dropout(x))
        return torch.sigmoid(raw) * self.max_values


class FusionHeads(nn.Module):
    """Bundles both heads over the fused representation. `forward` returns
    (class_logits, class_probs, score_preds); uncertainty and consistency
    are computed by the caller (src/train/train_fusion.py,
    src/explain/conformal.py) using the pieces exposed here.
    """

    def __init__(self, embed_dim: int = 256, classification_dropout: float = 0.3, regression_dropout: float = 0.3):
        super().__init__()
        self.classification_head = ClassificationHead(embed_dim, dropout=classification_dropout)
        self.regression_head = RegressionHead(embed_dim, dropout=regression_dropout)

    def forward(self, fused_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        class_logits = self.classification_head(fused_embedding)
        class_probs = torch.softmax(class_logits, dim=-1)
        score_preds = self.regression_head(fused_embedding)
        return class_logits, class_probs, score_preds
