"""Augmentation and class-imbalance handling for the three modalities.

Imbalance this module exists to address (docs/Dataset_Description.docx,
docs/MINDSCOPE_Blueprint.pdf Sec. 02), confirmed against the real data on disk:
  - FER facial: Disgust 436 vs Happy 7215 (~16.5x minority class)
  - RAVDESS speech: Neutral 96 vs the other 7 classes at 192 each (no "strong"
    intensity is recorded for neutral)
  - Tabular: Severe_Stress 128 vs Healthy 1629 (~12.7x minority class)

Augmentation is **train-split only**. Every function here is called from a
Dataset constructed with `train=True` (src/data/loaders.py); the eval/val
path uses `facial_eval_transform()` / no waveform augmentation / no SMOTE, so
validation metrics are measured on untouched data. tests/test_augment.py and
tests/test_loaders.py assert this separation holds.
"""
from __future__ import annotations

import numpy as np

# Minority classes that receive the stronger, targeted augmentation pipeline
# rather than the mild default (docs/MINDSCOPE_Blueprint.pdf Sec. 02:
# "targeted augmentation ... not accuracy chasing that ignores the rare class").
FACIAL_MINORITY_CLASSES = ("Disgust",)
SPEECH_MINORITY_CLASSES = ("neutral",)


# --------------------------------------------------------------------------
# Modality A -- facial images (albumentations)
# --------------------------------------------------------------------------
def facial_train_transform():
    """Mild albumentations pipeline applied to every FER training image.

    Kept deliberately gentle: 48x48 faces are already tightly registered, so
    aggressive geometry destroys the eye/mouth structure the CBAM attention
    and Grad-CAM depend on.
    """
    import albumentations as A

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-10, 10), translate_percent=0.05, scale=(0.95, 1.05), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussNoise(p=0.2),
        ]
    )


def facial_minority_transform():
    """Stronger targeted pipeline for the minority facial classes (Disgust,
    436 images vs Happy's 7215).

    This is the augmentation set named in the build brief -- horizontal flip,
    brightness/contrast jitter, shift-scale-rotate, and Gaussian blur -- with
    wider magnitudes and higher probabilities than the majority-class
    pipeline, so the rare class is effectively resampled through more
    diverse views instead of merely repeated.

    Note on the albumentations 2.x API: `ShiftScaleRotate` is deprecated in
    favour of `A.Affine`, which is what it now forwards to internally. We
    call `A.Affine` directly with shift (`translate_percent`), scale, and
    rotate limits so the pipeline is the same transform without the
    deprecation warning.
    """
    import albumentations as A

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
            A.Affine(  # == ShiftScaleRotate(shift=0.1, scale=0.15, rotate=20)
                translate_percent=(-0.10, 0.10),
                scale=(0.85, 1.15),
                rotate=(-20, 20),
                p=0.7,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        ]
    )


def facial_eval_transform():
    """No augmentation at eval time -- validation/test images pass through raw."""
    import albumentations as A

    return A.Compose([])


# --------------------------------------------------------------------------
# Modality B -- speech waveforms (librosa / torchaudio)
# --------------------------------------------------------------------------
def random_pitch_shift(waveform: np.ndarray, sample_rate: int, max_steps: float = 2.0) -> np.ndarray:
    """Random pitch shift of +/- `max_steps` semitones (librosa), leaving
    duration unchanged. Emotion in speech is carried heavily by prosody, so
    the shift is kept small -- a large shift changes perceived arousal and
    would relabel the sample rather than augment it.

    `res_type="soxr_lq"` rather than librosa's default `soxr_hq`: pitch
    shifting dominates the speech dataloader's cost (measured ~64 ms/sample
    vs ~2 ms unaugmented, i.e. 30x), which starves the GPU. The low-quality
    soxr kernel cuts that substantially, and its resampling artefacts sit far
    below the noise floor this augmentation deliberately adds anyway. (The
    `kaiser_*` kernels would need `resampy`, an extra dependency; soxr ships
    with librosa.)
    """
    import librosa

    steps = float(np.random.uniform(-max_steps, max_steps))
    if abs(steps) < 1e-3:
        return waveform
    return librosa.effects.pitch_shift(
        y=waveform, sr=sample_rate, n_steps=steps, res_type="soxr_lq"
    )


def inject_background_noise(waveform: np.ndarray, snr_db_range: tuple[float, float] = (15.0, 30.0)) -> np.ndarray:
    """Add Gaussian background noise at a random SNR in `snr_db_range`.

    Scaled relative to this clip's own power so the resulting SNR is the
    requested one regardless of the clip's absolute loudness.
    """
    signal_power = float(np.mean(waveform**2))
    if signal_power <= 0:
        return waveform
    snr_db = float(np.random.uniform(*snr_db_range))
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=waveform.shape).astype(waveform.dtype)
    return waveform + noise


def speech_train_augment(
    waveform,
    sample_rate: int,
    pitch_shift_p: float = 0.5,
    noise_p: float = 0.5,
    max_pitch_steps: float = 2.0,
):
    """Waveform-domain training augmentation: random pitch shift + background
    noise injection. Accepts and returns a torch tensor (the dtype
    SpeechEmotionDataset works in), converting to numpy only for librosa.

    SpecAugment (time/frequency masking) is deliberately NOT applied here --
    it operates on the log-Mel spectrogram, which is computed inside
    CNNBiLSTMEncoder, so it lives there (`spec_augment_torch`) and is
    likewise gated on `self.training`.
    """
    import torch

    array = waveform.detach().cpu().numpy().astype(np.float32)

    if np.random.rand() < pitch_shift_p:
        array = random_pitch_shift(array, sample_rate, max_steps=max_pitch_steps)
    if np.random.rand() < noise_p:
        array = inject_background_noise(array)

    return torch.from_numpy(np.ascontiguousarray(array)).float()


def spec_augment(mel_spectrogram: np.ndarray, freq_mask_param: int = 15, time_mask_param: int = 25):
    """SpecAugment (Park et al. 2019) frequency + time masking for log-Mel
    spectrograms, numpy version. `mel_spectrogram`: (n_mels, n_frames).
    """
    spec = mel_spectrogram.copy()
    n_mels, n_frames = spec.shape

    f = np.random.randint(0, freq_mask_param)
    f0 = np.random.randint(0, max(1, n_mels - f))
    spec[f0 : f0 + f, :] = spec.mean()

    t = np.random.randint(0, time_mask_param)
    t0 = np.random.randint(0, max(1, n_frames - t))
    spec[:, t0 : t0 + t] = spec.mean()

    return spec


def spec_augment_torch(spec, freq_mask_param: int = 15, time_mask_param: int = 25, n_masks: int = 1):
    """Batched SpecAugment for a torch log-Mel tensor of shape
    (B, n_mels, n_frames), applied inside the speech encoder's forward pass
    when (and only when) the module is in training mode.

    Uses torchaudio's FrequencyMasking/TimeMasking so masking runs on-device
    without a CPU round-trip.
    """
    import torchaudio

    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
    for _ in range(n_masks):
        spec = freq_mask(spec)
        spec = time_mask(spec)
    return spec


# --------------------------------------------------------------------------
# Modality C -- tabular (imbalanced-learn SMOTE)
# --------------------------------------------------------------------------
def tabular_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42, k_neighbors: int = 5):
    """SMOTE oversampling for the tabular FT-Transformer path
    (docs/MINDSCOPE_Blueprint.pdf Sec. 04, configs/tabular.yaml
    `imbalance.strategy: smote`).

    Must be fit on the TRAIN SPLIT ONLY -- synthesising minority rows before
    splitting leaks interpolated neighbours of validation rows into training
    and inflates every metric. `src/train/train_modality.py` calls this after
    the split; tests/test_augment.py pins that ordering.

    `k_neighbors` is clamped to the smallest class count - 1, since SMOTE
    raises if a class has fewer samples than neighbours requested (the real
    Severe_Stress class has 128 rows, so the default 5 is safe, but a
    stratified subsample used in tests may not be).
    """
    from imblearn.over_sampling import SMOTE

    _classes, counts = np.unique(y, return_counts=True)
    k = int(min(k_neighbors, counts.min() - 1))
    if k < 1:
        # A class with a single member cannot be interpolated -- return the
        # data untouched rather than raising, and let class-balanced focal
        # loss carry the imbalance handling on its own.
        return X, y

    smote = SMOTE(random_state=random_state, k_neighbors=k)
    return smote.fit_resample(X, y)


# --------------------------------------------------------------------------
# Sampling-based imbalance strategies (shared)
# --------------------------------------------------------------------------
def oversample_minority_indices(labels: np.ndarray, target_count: int | None = None) -> np.ndarray:
    """Return an index array that oversamples minority classes up to
    `target_count` (defaults to the majority class count), for use with
    torch's SubsetRandomSampler as a SMOTE-lite alternative for image data
    (SMOTE itself operates on feature vectors, not raw pixels -- see
    `tabular_smote` for the FT-Transformer path).
    """
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if target_count is None:
        target_count = int(counts.max())

    indices = []
    for cls, count in zip(classes, counts):
        cls_indices = np.where(labels == cls)[0]
        if count >= target_count:
            indices.append(cls_indices)
        else:
            repeats = target_count // count
            remainder = target_count % count
            oversampled = np.concatenate(
                [np.tile(cls_indices, repeats), np.random.choice(cls_indices, remainder, replace=False)]
            )
            indices.append(oversampled)
    result = np.concatenate(indices)
    np.random.shuffle(result)
    return result


def class_balanced_sample_weights(labels: np.ndarray, beta: float = 0.999) -> np.ndarray:
    """Class-Balanced sampling weights (Cui et al. 2019), used to build a
    torch WeightedRandomSampler as the default (non-oversampling) imbalance
    strategy for both FER and RAVDESS.
    """
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    effective_num = 1.0 - np.power(beta, counts)
    class_weights = (1.0 - beta) / effective_num
    class_weights = class_weights / class_weights.sum() * len(classes)
    weight_by_class = dict(zip(classes, class_weights))
    return np.array([weight_by_class[label] for label in labels], dtype=np.float32)
