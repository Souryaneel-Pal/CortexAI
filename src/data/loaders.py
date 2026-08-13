"""PyTorch Dataset classes for the three CortexAI modalities.

On-disk layout under data/raw/ as the Hack4Health archive ships it (see the
"Dataset Access" section of README.md):

    data/raw/
      Extracted_images/<EmotionName>/*.png   # 28,709 FER-style 48x48 grayscale
      Audios/Actor_XX/*.wav                  # 1,440 RAVDESS clips, 7-part filenames
      mental_health_multimodal.csv           # 4,000 rows, 18 features + 4 targets

Two quirks of the distributed archive are handled here rather than requiring
manual cleanup:
  - `Audios/` also contains a nested `audio_speech_actors_01-24/` directory
    holding a byte-identical copy of every Actor_XX folder. `SpeechEmotionDataset`
    walks recursively and de-duplicates by RAVDESS filename, so it loads 1,440
    clips from either layout instead of 2,880 duplicates.
  - The facial folders are also accepted as a single `fer2013.csv`.

Augmentation is train-split only: pass `train=True` to get the augmented
pipeline (src/data/augment.py), leave it False (the default) for validation
and test so metrics are measured on untouched data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.data.augment import (
    FACIAL_MINORITY_CLASSES,
    SPEECH_MINORITY_CLASSES,
    facial_eval_transform,
    facial_minority_transform,
    facial_train_transform,
    speech_train_augment,
)
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

# Canonical dataset locations (the paths the archive actually uses).
DEFAULT_FACIAL_ROOT = "data/raw/Extracted_images"
DEFAULT_SPEECH_ROOT = "data/raw/Audios"
DEFAULT_TABULAR_CSV = "data/raw/mental_health_multimodal.csv"

FACIAL_EMOTION_NAME_TO_IDX = {name: idx for idx, name in FACIAL_EMOTIONS.items()}
SPEECH_EMOTION_NAME_TO_IDX = {name: idx for idx, name in enumerate(SPEECH_EMOTIONS.values())}

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def _require_path(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{hint} not found at {path}. Download the dataset from the Google Drive link in "
            "README.md ('Dataset Access') and place it at this path, then re-run "
            "`python -m src.data.validate_datasets`."
        )
    return path


class FacialEmotionDataset(Dataset):
    """FER-style 48x48 grayscale facial emotion dataset.

    Supports two layouts:
      1. <root>/<EmotionName>/*.png  (one subfolder per class -- the shipped layout)
      2. <root>/fer2013.csv          (columns: emotion,pixels[,Usage])

    With `train=True`, minority classes (Disgust: 436 images vs Happy's 7,215)
    get the stronger targeted albumentations pipeline while the majority
    classes get the mild one, so the rare class is resampled through diverse
    views rather than simply repeated. With `train=False` no augmentation is
    applied at all.
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_FACIAL_ROOT,
        transform=None,
        train: bool = False,
    ):
        self.root = Path(root)
        self.train = train
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
                    if img_path.suffix.lower() in _IMAGE_SUFFIXES:
                        self.samples.append((img_path, label_idx))
            if not self.samples:
                raise FileNotFoundError(
                    f"No facial images found under {self.root}/<EmotionName>/ and no "
                    f"fer2013.csv present. See README.md 'Dataset Access'."
                )

        # Explicit transform wins; otherwise pick by train/eval mode. The
        # minority pipeline is selected per-sample in __getitem__.
        self._explicit_transform = transform
        self._majority_transform = facial_train_transform() if train else None
        self._minority_transform = facial_minority_transform() if train else None
        self._eval_transform = facial_eval_transform() if not train else None
        self._minority_label_indices = {
            FACIAL_EMOTION_NAME_TO_IDX[name]
            for name in FACIAL_MINORITY_CLASSES
            if name in FACIAL_EMOTION_NAME_TO_IDX
        }

    @property
    def labels(self) -> np.ndarray:
        """All label indices, for building class-balanced samplers."""
        return np.array([label for _source, label in self.samples], dtype=np.int64)

    def _transform_for(self, label: int):
        if self._explicit_transform is not None:
            return self._explicit_transform
        if not self.train:
            return self._eval_transform
        if label in self._minority_label_indices:
            return self._minority_transform
        return self._majority_transform

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

        transform = self._transform_for(label)
        if transform is not None:
            image = transform(image=image)["image"]

        # Normalize to a (1, H, W) float32 tensor in [0, 1] so DataLoader's
        # default collate produces a proper (B, 1, H, W) batch straight into
        # FaceEmotionEncoder -- raw numpy/uint8 arrays would either fail to
        # collate or feed the model unnormalized pixel values.
        image = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0) / 255.0
        return image, label


class SpeechEmotionDataset(Dataset):
    """RAVDESS speech-emotion dataset. One .wav per sample, label parsed from
    the 7-part filename.

    Walks `root` recursively and de-duplicates by filename, so the archive's
    duplicated `audio_speech_actors_01-24/` copy yields 1,440 clips rather
    than 2,880 (see module docstring).

    Waveforms are resampled to `sample_rate` and pad/truncated to a fixed
    `max_duration_sec` so DataLoader's default collate can batch them --
    RAVDESS clips are naturally variable-length.

    With `train=True`, waveform augmentation (random pitch shift + background
    noise injection, src/data/augment.py) is applied. SpecAugment's
    time/frequency masking happens later, inside CNNBiLSTMEncoder's forward
    pass, since it operates on the log-Mel spectrogram.
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_SPEECH_ROOT,
        transform=None,
        sample_rate: int = 16000,
        max_duration_sec: float = 4.0,
        train: bool = False,
    ):
        self.root = _require_path(Path(root), "Speech dataset root")
        self.transform = transform
        self.train = train
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration_sec)

        by_name: dict[str, tuple[Path, dict]] = {}
        for wav_path in sorted(self.root.rglob("*.wav")):
            if wav_path.name in by_name:
                continue  # duplicate copy of an already-seen clip
            try:
                meta = parse_ravdess_filename(wav_path.name)
            except (ValueError, KeyError):
                continue  # not a RAVDESS-convention filename -- skip rather than crash
            by_name[wav_path.name] = (wav_path, meta)

        self.samples: list[tuple[Path, dict]] = [by_name[name] for name in sorted(by_name)]
        if not self.samples:
            raise FileNotFoundError(
                f"No RAVDESS-named .wav files found under {self.root}. See README.md 'Dataset Access'."
            )

        self._minority_emotions = set(SPEECH_MINORITY_CLASSES)

    @property
    def labels(self) -> np.ndarray:
        """All emotion label indices, for building class-balanced samplers."""
        return np.array(
            [SPEECH_EMOTION_NAME_TO_IDX[meta["emotion_label"]] for _p, meta in self.samples],
            dtype=np.int64,
        )

    @property
    def actor_ids(self) -> np.ndarray:
        """Actor id per sample -- speech splits are grouped by actor to avoid
        speaker leakage between train and val (configs/speech.yaml `split_by`).
        """
        return np.array([meta["actor_id"] for _p, meta in self.samples], dtype=np.int64)

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
        elif self.train:
            # Minority class (neutral: 96 clips vs 192 for every other
            # emotion) gets a higher augmentation rate and a wider pitch
            # range, matching the facial minority strategy.
            is_minority = meta["emotion_label"] in self._minority_emotions
            waveform = speech_train_augment(
                waveform,
                self.sample_rate,
                pitch_shift_p=0.8 if is_minority else 0.5,
                noise_p=0.8 if is_minority else 0.5,
                max_pitch_steps=3.0 if is_minority else 2.0,
            )

        if waveform.shape[0] >= self.max_samples:
            waveform = waveform[: self.max_samples]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_samples - waveform.shape[0]))

        label = SPEECH_EMOTION_NAME_TO_IDX[meta["emotion_label"]]
        return waveform.float(), label, meta


class TabularMentalHealthDataset(Dataset):
    """The 4000-row, 18-feature labelled table -- the only source of ground truth.

    `labels` are integer StressLevel indices (0=Healthy .. 3=Severe) so the
    dataset collates straight into a loss function; the raw string labels
    stay available on `.df` for class-count/fairness reporting.
    """

    def __init__(
        self,
        csv_path: str | Path = DEFAULT_TABULAR_CSV,
        scaler=None,
        label_encoder=None,
        indices: np.ndarray | None = None,
    ):
        self.csv_path = _require_path(Path(csv_path), "Tabular dataset CSV")
        self.df = pd.read_csv(self.csv_path)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)

        missing = set(TABULAR_FEATURE_COLUMNS) - set(self.df.columns)
        if missing:
            raise ValueError(f"Tabular CSV missing expected columns: {sorted(missing)}")

        self.features = self.df[TABULAR_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        if scaler is not None:
            self.features = scaler.transform(self.features).astype(np.float32)

        raw_labels = self.df[TABULAR_TARGET_CLASS_COLUMN].to_numpy()
        if label_encoder is not None:
            self.labels = label_encoder.transform(raw_labels)
        else:
            self.labels = np.array(
                [int(STRESS_LABEL_NAME_TO_LEVEL[name]) for name in raw_labels], dtype=np.int64
            )
        self.raw_labels = raw_labels

        self.scores = self.df[TABULAR_TARGET_SCORE_COLUMNS].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx], self.scores[idx]


class FusionPairDataset(Dataset):
    """Anchored, weakly-paired dataset for fusion training
    (PROJECT_PLAN.md P2, docs/MINDSCOPE_Blueprint.pdf Sec. 02 "STAGE 3").

    The tabular dataset is the only source of real ground truth. This class
    does NOT invent face+voice+row triples: for each tabular row it samples a
    facial image and a speech clip, and in `pair_by_label=True` mode it
    restricts that draw to media whose OWN native emotion label maps (via the
    modality-specific tables in src/data/emotion_stress_map.py) onto the same
    stress tier as the row's real Mental_Health_Status.

    IMPORTANT -- why validation must use `pair_by_label=False`:
    matched-emotion pairing keys the sampled media on the row's ground-truth
    label, so the face and voice inputs *encode the answer*. That is a
    defensible training-time prior (it teaches the fusion gate what
    tier-consistent evidence looks like), but scoring a validation split that
    was paired the same way measures the model's ability to read the leak,
    not to assess a person -- it inflates fusion metrics arbitrarily. So
    `src/train/train_fusion.py` builds the train split with
    `pair_by_label=True` and the val split with `pair_by_label=False`
    (media drawn uniformly at random, exactly like a real session where the
    label is unknown). tests/test_fusion_pairing.py pins this.

    Call `resample()` once per epoch (configs/fusion.yaml
    `resample_pairs_every_epoch: true`) to draw a fresh random pairing so the
    model doesn't overfit to one arbitrary face/voice per row.
    """

    def __init__(
        self,
        tabular_dataset: TabularMentalHealthDataset,
        facial_dataset: FacialEmotionDataset,
        speech_dataset: SpeechEmotionDataset,
        seed: int = 42,
        pair_by_label: bool = True,
    ):
        self.tabular_dataset = tabular_dataset
        self.facial_dataset = facial_dataset
        self.speech_dataset = speech_dataset
        self.pair_by_label = pair_by_label
        self._rng = np.random.default_rng(seed)

        self._facial_indices_by_tier = self._index_by_stress_tier(
            facial_dataset, FACIAL_EMOTION_NAME_TO_IDX, FACIAL_EMOTION_TO_STRESS
        )
        self._speech_indices_by_tier = self._index_speech_by_stress_tier(speech_dataset)

        if pair_by_label:
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
        """Draw a fresh face/speech pairing for every tabular row."""
        n = len(self.tabular_dataset)
        labels = self.tabular_dataset.labels

        if not self.pair_by_label:
            # Label-independent pairing -- the honest evaluation/inference
            # regime (see class docstring).
            self._facial_pairs = self._rng.integers(0, len(self.facial_dataset), size=n)
            self._speech_pairs = self._rng.integers(0, len(self.speech_dataset), size=n)
            return

        facial_pairs = np.empty(n, dtype=np.int64)
        speech_pairs = np.empty(n, dtype=np.int64)
        for i in range(n):
            tier = int(labels[i])
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
