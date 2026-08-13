"""Gated cross-modal attention fusion (docs/MINDSCOPE_Blueprint.pdf Sec. 03-04).

Operates purely at the embedding level: each per-modality encoder (P1) emits
a 256-d embedding; this module fuses the three (face, speech, tabular) into
one joint representation via a small Transformer, with a learned gate that
scores each modality's reliability per-sample -- doubling as the "modality
contribution meter" for Objective 3 (e.g. "72% physiological, 20% facial,
8% vocal").

Trained with modality dropout (a modality's embedding is replaced with a
learned "missing" token during training) so the model degrades gracefully
when a modality is unavailable at inference -- essential for real clinical
use where a webcam/mic/wearable may be missing.
"""
from __future__ import annotations

import torch
import torch.nn as nn

MODALITIES = ("face", "speech", "tabular")
EMBED_DIM = 256


class ModalityDropout(nn.Module):
    """During training, independently zeroes out each modality's embedding
    with probability `p` and substitutes a learned "missing modality" token,
    so the fusion gate learns to redistribute weight onto the remaining
    modalities rather than relying on all three always being present.
    At eval time this is a no-op unless `modality_mask` explicitly marks a
    modality missing (real missing-modality inference, e.g. no webcam).
    """

    def __init__(self, embed_dim: int = EMBED_DIM, p: float = 0.15):
        super().__init__()
        self.p = p
        self.missing_tokens = nn.ParameterDict(
            {m: nn.Parameter(torch.randn(embed_dim) * 0.02) for m in MODALITIES}
        )

    def forward(
        self, embeddings: dict[str, torch.Tensor], modality_mask: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        out = {}
        for modality, embedding in embeddings.items():
            batch_size = embedding.shape[0]
            missing_token = self.missing_tokens[modality].unsqueeze(0).expand(batch_size, -1)

            if modality_mask is not None and modality in modality_mask:
                # Explicit availability mask (1 = present, 0 = missing) -- the
                # real-world "no webcam this session" case, applies at both
                # train and eval time.
                present = modality_mask[modality].bool().unsqueeze(-1)
                embedding = torch.where(present, embedding, missing_token)

            if self.training and self.p > 0:
                drop_mask = (torch.rand(batch_size, 1, device=embedding.device) < self.p)
                embedding = torch.where(drop_mask, missing_token, embedding)

            out[modality] = embedding
        return out


class GatedCrossModalFusion(nn.Module):
    """Three 256-d modality embeddings -> gated cross-modal attention ->
    one fused 256-d representation + per-modality contribution weights.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        modality_dropout_p: float = 0.15,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.modality_dropout = ModalityDropout(embed_dim, p=modality_dropout_p)

        self.input_proj = nn.ModuleDict(
            {m: nn.Linear(embed_dim, hidden_dim) for m in MODALITIES}
        )
        self.modality_type_embed = nn.ParameterDict(
            {m: nn.Parameter(torch.randn(hidden_dim) * 0.02) for m in MODALITIES}
        )
        self.fusion_cls = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # Gate: scores each modality's reliability from its own (post-dropout)
        # embedding, conditioned on the others -- the modality-contribution
        # meter exposed to src/explain/ for Objective 3.
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * len(MODALITIES), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(MODALITIES)),
        )

        self.output_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(
        self,
        face_embedding: torch.Tensor,
        speech_embedding: torch.Tensor,
        tabular_embedding: torch.Tensor,
        modality_mask: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (fused_embedding[B, embed_dim], modality_weights[B, 3]),
        with modality_weights index-aligned to MODALITIES = (face, speech, tabular).
        """
        embeddings = {"face": face_embedding, "speech": speech_embedding, "tabular": tabular_embedding}
        embeddings = self.modality_dropout(embeddings, modality_mask)

        tokens = []
        for modality in MODALITIES:
            projected = self.input_proj[modality](embeddings[modality])
            projected = projected + self.modality_type_embed[modality]
            tokens.append(projected)
        stacked = torch.stack(tokens, dim=1)  # (B, 3, hidden_dim)

        gate_logits = self.gate(stacked.flatten(start_dim=1))  # (B, 3)
        modality_weights = torch.softmax(gate_logits, dim=-1)
        gated_tokens = stacked * modality_weights.unsqueeze(-1)

        batch_size = gated_tokens.shape[0]
        cls = self.fusion_cls.expand(batch_size, -1, -1)
        sequence = torch.cat([cls, gated_tokens], dim=1)  # (B, 4, hidden_dim)

        encoded = self.encoder(sequence)
        fused = self.norm(encoded[:, 0, :])  # pooled [CLS]
        fused = self.output_proj(fused)
        return fused, modality_weights
