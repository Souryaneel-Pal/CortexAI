"""Facial emotion encoder (docs/MINDSCOPE_Blueprint.pdf Sec. 04).

Primary: EfficientNet-B0 (timm) + CBAM attention, fine-tuned on FER, exposing
a 256-d embedding and 7-way emotion logits. CBAM focuses the network on the
expressive regions (eyes/mouth), which also yields cleaner Grad-CAM heatmaps
for Objective 3.

Fallback (used when `backbone: simple_cnn` in configs/face.yaml -- e.g. if
timm/pretrained-weight download is blocked): a 4-block CNN from scratch.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.data.schemas import FACIAL_EMOTIONS

NUM_FACIAL_EMOTIONS = len(FACIAL_EMOTIONS)
EMBED_DIM = 256


class ChannelAttention(nn.Module):
    """CBAM channel-attention submodule (Woo et al. 2018)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return x * attn


class SpatialAttention(nn.Module):
    """CBAM spatial-attention submodule."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention then spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SimpleCNN(nn.Module):
    """4-block CNN-from-scratch fallback for the facial encoder."""

    FEATURE_CHANNELS = 256

    def __init__(self, in_channels: int = 1):
        super().__init__()
        blocks = []
        channels = [in_channels, 32, 64, 128, 256]
        for i in range(4):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.cbam = CBAM(channels[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.cbam(x)
        return x  # (B, 256, H', W')


class EfficientNetB0CBAM(nn.Module):
    """EfficientNet-B0 backbone (timm) with a CBAM block inserted after the
    final feature map, before global pooling.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            in_chans=1,  # FER is grayscale
            features_only=True,
            out_indices=(4,),  # last feature stage
        )
        feature_channels = self.backbone.feature_info.channels()[-1]
        self.cbam = CBAM(feature_channels)
        self.out_channels = feature_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)[-1]
        return self.cbam(features)  # (B, C, H', W')


class FaceEmotionEncoder(nn.Module):
    """Full facial encoder: backbone + CBAM -> pooled 256-d embedding -> 7-way
    emotion head. `forward` returns (embedding, logits) so downstream fusion
    (src/models/fusion.py) consumes the embedding and P1 training consumes
    the logits directly.

    Grad-CAM (src/explain/gradcam.py) hooks the backbone's last conv block --
    see `gradcam_target_layer`.
    """

    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "efficientnet_b0":
            self.feature_extractor = EfficientNetB0CBAM(pretrained=pretrained)
            feature_channels = self.feature_extractor.out_channels
        elif backbone == "simple_cnn":
            self.feature_extractor = SimpleCNN()
            feature_channels = SimpleCNN.FEATURE_CHANNELS
        else:
            raise ValueError(f"Unknown facial backbone: {backbone!r}")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_channels, EMBED_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.emotion_head = nn.Linear(EMBED_DIM, NUM_FACIAL_EMOTIONS)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """The conv layer Grad-CAM should hook -- the last convolutional block
        of the EfficientNet-B0 backbone (or the CBAM block for simple_cnn).
        """
        if self.backbone_name == "efficientnet_b0":
            return self.feature_extractor.backbone.blocks[6]
        else:
            return self.feature_extractor.cbam

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 1, 48, 48) grayscale batch. Returns (embedding[B,256], logits[B,7])."""
        features = self.feature_extractor(x)
        embedding = self.embed_proj(self.pool(features))
        logits = self.emotion_head(embedding)
        return embedding, logits
