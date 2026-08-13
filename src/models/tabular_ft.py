"""Tabular encoder (docs/MINDSCOPE_Blueprint.pdf Sec. 04).

Primary: FT-Transformer (Feature Tokenizer + Transformer, Gorishniy et al.
2021) over the 18 numeric features, exposing a 256-d embedding. Per-feature
attention doubles as a feature-importance signal that feeds Objective 3
(alongside SHAP, src/explain/shap_tab.py). Stacked with LightGBM
(`LightGBMStack` below) since gradient-boosted trees usually win on raw
tabular accuracy -- we don't hide that trade-off, we use both honestly.

Fallback (used when `backbone: residual_mlp` in configs/tabular.yaml):
a residual MLP + BatchNorm.

This is the only modality with real ground truth, so this module also
exposes the joint 4-class + 3-score heads directly (P1 baseline trains them
standalone; P2 fusion reuses the embedding with the shared fusion heads in
src/models/heads.py).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.data.schemas import (
    StressLevel,
    TABULAR_FEATURE_COLUMNS,
    TABULAR_TARGET_SCORE_COLUMNS,
)

NUM_FEATURES = len(TABULAR_FEATURE_COLUMNS)
NUM_CLASSES = len(StressLevel)
NUM_SCORE_TARGETS = len(TABULAR_TARGET_SCORE_COLUMNS)
EMBED_DIM = 256


class FeatureTokenizer(nn.Module):
    """Turns each of the 18 scalar features into its own `token_dim`-wide
    token (Gorishniy et al. 2021): x_i -> w_i * x_i + b_i, plus a learned
    [CLS] token prepended for pooling.
    """

    def __init__(self, num_features: int, token_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, token_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, token_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, num_features) -> (B, num_features + 1, token_dim)."""
        tokens = x.unsqueeze(-1) * self.weight + self.bias  # (B, F, token_dim)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1)


class FTTransformerEncoder(nn.Module):
    """Feature Tokenizer + a standard pre-norm Transformer encoder, pooled at
    the [CLS] token into a 256-d embedding.
    """

    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        token_dim: int = 192,
        n_blocks: int = 3,
        n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        embed_dim: int = EMBED_DIM,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, token_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 4,
            dropout=ffn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.norm = nn.LayerNorm(token_dim)
        self.embed_proj = nn.Linear(token_dim, embed_dim)

        self._last_attention_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, num_features) standardized features -> (B, embed_dim)."""
        tokens = self.tokenizer(x)
        tokens = self.attention_dropout(tokens)
        encoded = self.encoder(tokens)
        cls_out = self.norm(encoded[:, 0, :])  # pooled [CLS] representation
        return self.embed_proj(cls_out)

    @torch.no_grad()
    def feature_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Per-feature attention weight from the [CLS] token to every feature
        token in the first encoder block -- a free feature-importance signal
        (Objective 3), independent of but consistent with SHAP.
        Returns (B, num_features).
        """
        tokens = self.tokenizer(x)
        first_layer = self.encoder.layers[0]
        attn_out, attn_weights = first_layer.self_attn(
            tokens, tokens, tokens, need_weights=True, average_attn_weights=True
        )
        # attn_weights: (B, seq_len, seq_len); row 0 = CLS attending to all tokens.
        return attn_weights[:, 0, 1:]  # drop attention-to-self(CLS), keep the 18 features


class ResidualMLPEncoder(nn.Module):
    """Residual MLP + BatchNorm fallback tabular encoder."""

    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
        embed_dim: int = EMBED_DIM,
    ):
        super().__init__()
        dims = [num_features, *hidden_dims]
        blocks = []
        for i in range(len(hidden_dims)):
            blocks.append(
                nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.BatchNorm1d(dims[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.residual_proj = nn.ModuleList(
            [
                nn.Linear(dims[i], dims[i + 1]) if dims[i] != dims[i + 1] else nn.Identity()
                for i in range(len(hidden_dims))
            ]
        )
        self.embed_proj = nn.Linear(dims[-1], embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block, proj in zip(self.blocks, self.residual_proj):
            x = block(x) + proj(x)
        return self.embed_proj(x)


class TabularEncoder(nn.Module):
    """Wraps the chosen backbone with the P1 baseline heads (4-class + 3-score),
    trained standalone on the real tabular ground truth.
    """

    def __init__(
        self,
        backbone: str = "ft_transformer",
        num_features: int = NUM_FEATURES,
        embed_dim: int = EMBED_DIM,
        dropout: float = 0.3,
        **backbone_kwargs,
    ):
        super().__init__()
        if backbone == "ft_transformer":
            self.encoder: nn.Module = FTTransformerEncoder(
                num_features=num_features, embed_dim=embed_dim, **backbone_kwargs
            )
        elif backbone == "residual_mlp":
            self.encoder = ResidualMLPEncoder(
                num_features=num_features, embed_dim=embed_dim, dropout=dropout, **backbone_kwargs
            )
        else:
            raise ValueError(f"Unknown tabular backbone: {backbone!r}")

        self.classification_head = nn.Linear(embed_dim, NUM_CLASSES)
        self.regression_head = nn.Linear(embed_dim, NUM_SCORE_TARGETS)

    def forward(self, x: torch.Tensor):
        """x: (B, 18) standardized features.
        Returns (embedding[B,256], class_logits[B,4], score_preds[B,3]).
        """
        embedding = self.encoder(x)
        class_logits = self.classification_head(embedding)
        score_preds = self.regression_head(embedding)
        return embedding, class_logits, score_preds


class LightGBMStack:
    """LightGBM classifier + per-target regressors stacked alongside the
    FT-Transformer, per the "honest engineering note" in
    docs/MINDSCOPE_Blueprint.pdf Sec. 04: trees usually win on raw tabular
    accuracy, so we ensemble rather than pretend the deep net alone is best.
    Not an nn.Module -- fit/predict operate on numpy arrays.
    """

    def __init__(self, lgbm_params: dict | None = None):
        import lightgbm as lgb

        params = lgbm_params or {}
        self.classifier = lgb.LGBMClassifier(**params)
        self.regressors = [lgb.LGBMRegressor(**params) for _ in range(NUM_SCORE_TARGETS)]

    def fit(self, X: np.ndarray, y_class: np.ndarray, y_scores: np.ndarray) -> "LightGBMStack":
        self.classifier.fit(X, y_class)
        for i, regressor in enumerate(self.regressors):
            regressor.fit(X, y_scores[:, i])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(X)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return np.stack([r.predict(X) for r in self.regressors], axis=1)


def ensemble_class_probs(ft_probs: np.ndarray, lgbm_probs: np.ndarray, ft_weight: float = 0.5) -> np.ndarray:
    """Simple weighted average of FT-Transformer and LightGBM class
    probabilities -- the "stack" in configs/tabular.yaml's
    `ensemble.use_lightgbm_stack`.
    """
    return ft_weight * ft_probs + (1 - ft_weight) * lgbm_probs
