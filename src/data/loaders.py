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

from src.data.emotion_stress_map import FACIAL_EMOTION_TO_STRESS, SPEECH_EMOTION_TO_STRESS
from src.data.schemas import (
    FACIAL_EMOTIONS,
    FACIAL_IMAGE_SIZE,
    SPEECH_EMOTIONS,
    STRESS_LABEL_NAME_TO_LEVEL,
    StressLevel,
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


class FusionPairDataset(Dataset):
    """Anchored, weakly-paired dataset for fusion training
    (docs/PROJECT_PLAN.md P2, docs/MINDSCOPE_Blueprint.pdf Sec. 02 "STAGE 3").

    The tabular dataset is the only source of real ground truth. This class
    does NOT invent face+voice+row triples: for each tabular row, it samples
    a facial image and a speech clip whose OWN native emotion label maps
    (via src/data/emotion_stress_map.py's modality-specific tables) onto the
    SAME stress tier the row's real Mental_Health_Status label carries. That
    is "weak-pairing via matched-emotion sampling" -- plausible stress-tier-
    consistent evidence, never a claimed identity match.

    Call `resample()` once per epoch (per configs/fusion.yaml's
    `resample_pairs_every_epoch: true`) to draw a fresh random pairing, so
    the model doesn't overfit to one arbitrary face/voice per row.
    """

    def __init__(
        self,
        tabular_dataset: TabularMentalHealthDataset,
        facial_dataset: FacialEmotionDataset,
        speech_dataset: SpeechEmotionDataset,
        seed: int = 42,
    ):
        self.tabular_dataset = tabular_dataset
        self.facial_dataset = facial_dataset
        self.speech_dataset = speech_dataset
        self._rng = np.random.default_rng(seed)

        self._facial_indices_by_tier = self._index_by_stress_tier(
            facial_dataset, FACIAL_EMOTION_NAME_TO_IDX, FACIAL_EMOTION_TO_STRESS
        )
        self._speech_indices_by_tier = self._index_speech_by_stress_tier(speech_dataset)

        for tier in StressLevel:
            if not self._facial_indices_by_tier.get(int(tier)):
                raise ValueError(f"No facial images map to stress tier {tier.name} -- cannot weak-pair.")
            if not self._speech_indices_by_tier.get(int(tier)):
                raise ValueError(f"No speech clips map to stress tier {tier.name} -- cannot weak-pair.")

        self.resample()

    @staticmethod
    def _index_by_stress_tier(facial_dataset, name_to_idx, emotion_to_stress) -> dict[int, list[int]]:
        idx_to_name = {v: k for k, v in name_to_idx.items()}
        buckets: dict[int, list[int]] = {int(t): [] for t in StressLevel}
        for sample_idx, (_source, label_idx) in enumerate(facial_dataset.samples):
            emotion_name = idx_to_name[label_idx]
            tier = emotion_to_stress[emotion_name]
            buckets[int(tier)].append(sample_idx)
        return buckets

    @staticmethod
    def _index_speech_by_stress_tier(speech_dataset) -> dict[int, list[int]]:
        buckets: dict[int, list[int]] = {int(t): [] for t in StressLevel}
        for sample_idx, (_path, meta) in enumerate(speech_dataset.samples):
            tier = SPEECH_EMOTION_TO_STRESS[meta["emotion_label"]]
            buckets[int(tier)].append(sample_idx)
        return buckets

    def resample(self) -> None:
        """Draw a fresh random face/speech pairing for every tabular row."""
        n = len(self.tabular_dataset)
        labels = self.tabular_dataset.labels
        facial_pairs = np.empty(n, dtype=np.int64)
        speech_pairs = np.empty(n, dtype=np.int64)
        for i in range(n):
            tier = int(STRESS_LABEL_NAME_TO_LEVEL[labels[i]])
            facial_pairs[i] = self._rng.choice(self._facial_indices_by_tier[tier])
            speech_pairs[i] = self._rng.choice(self._speech_indices_by_tier[tier])
        self._facial_pairs = facial_pairs
        self._speech_pairs = speech_pairs

    def __len__(self) -> int:
        return len(self.tabular_dataset)

    def __getitem__(self, idx: int):
        features, label, scores = self.tabular_dataset[idx]
        image, _facial_emotion = self.facial_dataset[self._facial_pairs[idx]]
        waveform, _speech_emotion, _meta = self.speech_dataset[self._speech_pairs[idx]]
        return features, label, scores, image, waveform
