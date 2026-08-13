"""PyTorch Dataset classes for the three CortexAI modalities.

Expected on-disk layout under data/raw/ (docs/1.docx, docs/MINDSCOPE_Blueprint.pdf
Sec. 10):

    data/raw/
      facial/<EmotionName>/*.png|jpg   # or a single fer2013.csv - both supported
      speech/*.wav                     # RAVDESS filenames, e.g. 03-01-06-01-02-01-12.wav
      numerical.csv                    # the 4000-row, 18-feature table

These loaders raise a clear FileNotFoundError with the expected path if the
data has not been supplied yet -- see PROJECT_PLAN.md P0 for the "data not
yet uploaded" status. They are otherwise fully functional and are exercised
against synthetic fixtures in tests/test_loaders.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.data.schemas import (
    FACIAL_EMOTIONS,
    FACIAL_IMAGE_SIZE,
    SPEECH_EMOTIONS,
    TABULAR_FEATURE_COLUMNS,
    TABULAR_TARGET_CLASS_COLUMN,
    TABULAR_TARGET_SCORE_COLUMNS,
    parse_ravdess_filename,
)

FACIAL_EMOTION_NAME_TO_IDX = {name: idx for idx, name in FACIAL_EMOTIONS.items()}
SPEECH_EMOTION_NAME_TO_IDX = {name: idx for idx, name in enumerate(SPEECH_EMOTIONS.values())}


def _require_path(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{hint} not found at {path}. Datasets have not been supplied yet -- "
            "see PROJECT_PLAN.md P0. Place the real data at this path and re-run."
        )
    return path


class FacialEmotionDataset(Dataset):
    """FER-style 48x48 grayscale facial emotion dataset.

    Supports two layouts:
      1. data/raw/facial/<EmotionName>/*.png (one subfolder per class)
      2. data/raw/facial/fer2013.csv (columns: emotion,pixels[,Usage])
    """

    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple] = []  # (image_source, label_idx)
        self._mode: str

        csv_path = self.root / "fer2013.csv"
        if csv_path.exists():
            self._mode = "csv"
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.samples.append((row["pixels"], int(row["emotion"])))
        else:
            _require_path(self.root, "Facial dataset root")
            self._mode = "folders"
            for emotion_name, label_idx in FACIAL_EMOTION_NAME_TO_IDX.items():
                class_dir = self.root / emotion_name
                if not class_dir.exists():
                    continue
                for img_path in sorted(class_dir.glob("*")):
                    if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        self.samples.append((img_path, label_idx))
            if not self.samples:
                raise FileNotFoundError(
                    f"No facial images found under {self.root}/<EmotionName>/ and no "
                    f"fer2013.csv present. Datasets have not been supplied yet."
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch

        source, label = self.samples[idx]
        if self._mode == "csv":
            pixels = np.array(source.split(), dtype=np.uint8)
            image = pixels.reshape(FACIAL_IMAGE_SIZE, FACIAL_IMAGE_SIZE)
        else:
            from PIL import Image

            image = np.array(Image.open(source).convert("L"))
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        # Normalize to a (1, H, W) float32 tensor in [0, 1] so DataLoader's
        # default collate produces a proper (B, 1, H, W) batch straight into
        # FaceEmotionEncoder -- raw numpy/uint8 arrays would either fail to
        # collate or feed the model unnormalized pixel values.
        image = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0) / 255.0
        return image, label


class SpeechEmotionDataset(Dataset):
    """RAVDESS speech-emotion dataset. One .wav per sample, label parsed from filename.

    Waveforms are resampled to `sample_rate` and pad/truncated to a fixed
    `max_duration_sec` so DataLoader's default collate can batch them --
    RAVDESS clips are naturally variable-length, which would otherwise fail
    to stack into a single tensor.
    """

    def __init__(
        self,
        root: str | Path,
        transform=None,
        sample_rate: int = 16000,
        max_duration_sec: float = 4.0,
    ):
        self.root = _require_path(Path(root), "Speech dataset root")
        self.transform = transform
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration_sec)
        self.samples: list[tuple[Path, dict]] = []
        for wav_path in sorted(self.root.glob("*.wav")):
            meta = parse_ravdess_filename(wav_path.name)
            self.samples.append((wav_path, meta))
        if not self.samples:
            raise FileNotFoundError(f"No .wav files found under {self.root}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import soundfile as sf
        import torch
        import torchaudio

        wav_path, meta = self.samples[idx]
        waveform, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        waveform = torch.from_numpy(np.atleast_1d(waveform))
        if waveform.ndim > 1:  # collapse multi-channel to mono
            waveform = waveform.mean(dim=-1)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        if self.transform is not None:
            waveform = self.transform(waveform, self.sample_rate)

        if waveform.shape[0] >= self.max_samples:
            waveform = waveform[: self.max_samples]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_samples - waveform.shape[0]))

        label = SPEECH_EMOTION_NAME_TO_IDX[meta["emotion_label"]]
        return waveform.float(), label, meta


class TabularMentalHealthDataset(Dataset):
    """The 4000-row, 18-feature labelled table -- the only source of ground truth."""

    def __init__(self, csv_path: str | Path, scaler=None, label_encoder=None):
        self.csv_path = _require_path(Path(csv_path), "Tabular dataset CSV")
        self.df = pd.read_csv(self.csv_path)

        missing = set(TABULAR_FEATURE_COLUMNS) - set(self.df.columns)
        if missing:
            raise ValueError(f"Tabular CSV missing expected columns: {sorted(missing)}")

        self.features = self.df[TABULAR_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        if scaler is not None:
            self.features = scaler.transform(self.features)

        raw_labels = self.df[TABULAR_TARGET_CLASS_COLUMN].to_numpy()
        if label_encoder is not None:
            self.labels = label_encoder.transform(raw_labels)
        else:
            self.labels = raw_labels

        self.scores = self.df[TABULAR_TARGET_SCORE_COLUMNS].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx], self.scores[idx]
