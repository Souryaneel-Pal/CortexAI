"""Speech emotion encoder (docs/MINDSCOPE_Blueprint.pdf Sec. 04).

Primary: Wav2Vec2-base (HuggingFace Transformers) fine-tuned for 8-way
RAVDESS emotion, exposing a 256-d embedding.

Fallback (used when `backbone: cnn_bilstm` in configs/speech.yaml -- e.g. if
wav2vec2 weight download is blocked, or fine-tuning is too slow on CPU):
CNN-BiLSTM over log-Mel spectrograms + SpecAugment (src/data/augment.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

from src.data.schemas import SPEECH_EMOTIONS

NUM_SPEECH_EMOTIONS = len(SPEECH_EMOTIONS)
EMBED_DIM = 256


class Wav2Vec2EmotionEncoder(nn.Module):
    """Wav2Vec2-base + a pooled projection head + emotion classifier."""

    def __init__(
        self,
        pretrained_checkpoint: str = "facebook/wav2vec2-base",
        freeze_feature_extractor: bool = True,
        dropout: float = 0.3,
        normalize_input: bool = True,
    ):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_checkpoint)
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()

        # facebook/wav2vec2-base ships with `do_normalize=True`: its
        # Wav2Vec2FeatureExtractor zero-means and unit-variances each waveform
        # before the model sees it. We feed waveforms straight from the
        # Dataset rather than through that processor, so the same
        # normalisation has to happen here -- without it the pretrained
        # features are fed off-distribution input and the model plateaus at
        # chance (measured: 13.3% accuracy, i.e. one constant class).
        self.normalize_input = normalize_input

        hidden_size = self.wav2vec2.config.hidden_size
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size, EMBED_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.emotion_head = nn.Linear(EMBED_DIM, NUM_SPEECH_EMOTIONS)

    def forward(self, waveform: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """waveform: (B, T) raw 16kHz audio. Returns (embedding[B,256], logits[B,8])."""
        if self.normalize_input:
            mean = waveform.mean(dim=-1, keepdim=True)
            std = waveform.std(dim=-1, keepdim=True)
            waveform = (waveform - mean) / (std + 1e-7)
        outputs = self.wav2vec2(waveform, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)  # mean pool over time
        embedding = self.embed_proj(pooled)
        logits = self.emotion_head(embedding)
        return embedding, logits


class CNNBiLSTMEncoder(nn.Module):
    """Log-Mel spectrogram -> CNN -> BiLSTM fallback speech encoder.

    A respected, lightweight SER baseline that trains in minutes on CPU
    (docs/MINDSCOPE_Blueprint.pdf Sec. 04) -- used when wav2vec2 is blocked.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        cnn_channels: tuple[int, ...] = (32, 64, 128),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        spec_augment: bool = True,
        freq_mask_param: int = 15,
        time_mask_param: int = 25,
    ):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=400, hop_length=160
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # SpecAugment (configs/speech.yaml `augment.spec_augment`) masks
        # time and frequency bands of the log-Mel spectrogram. It lives here
        # rather than in the Dataset because the spectrogram is computed in
        # this forward pass, and it is gated on `self.training` so validation
        # and inference always see unmasked spectrograms.
        self.spec_augment = spec_augment
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)

        channels = [1, *cnn_channels]
        conv_blocks = []
        for i in range(len(cnn_channels)):
            conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d((2, 1)),  # downsample freq axis only, preserve time
                )
            )
        self.conv_blocks = nn.ModuleList(conv_blocks)

        freq_dim = n_mels // (2 ** len(cnn_channels))
        lstm_input_size = cnn_channels[-1] * max(freq_dim, 1)
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.embed_proj = nn.Sequential(
            nn.Linear(lstm_out_dim, EMBED_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.emotion_head = nn.Linear(EMBED_DIM, NUM_SPEECH_EMOTIONS)

    def forward(self, waveform: torch.Tensor):
        """waveform: (B, T) raw audio. Returns (embedding[B,256], logits[B,8])."""
        spec = self.amplitude_to_db(self.melspec(waveform))  # (B, n_mels, n_frames)
        if self.spec_augment and self.training:
            spec = self.time_masking(self.freq_masking(spec))
        x = spec.unsqueeze(1)  # (B, 1, n_mels, n_frames)
        for block in self.conv_blocks:
            x = block(x)
        # (B, C, F', T') -> (B, T', C*F')
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        lstm_out, _ = self.lstm(x)
        pooled = lstm_out.mean(dim=1)
        embedding = self.embed_proj(pooled)
        logits = self.emotion_head(embedding)
        return embedding, logits


def build_speech_encoder(
    backbone: str = "wav2vec2",
    pretrained_checkpoint: str = "facebook/wav2vec2-base",
    freeze_feature_extractor: bool = True,
    **fallback_kwargs,
) -> nn.Module:
    if backbone == "wav2vec2":
        return Wav2Vec2EmotionEncoder(
            pretrained_checkpoint=pretrained_checkpoint,
            freeze_feature_extractor=freeze_feature_extractor,
        )
    if backbone == "cnn_bilstm":
        return CNNBiLSTMEncoder(**fallback_kwargs)
    raise ValueError(f"Unknown speech backbone: {backbone!r}")
