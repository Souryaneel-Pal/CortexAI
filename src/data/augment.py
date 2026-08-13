"""Augmentation and class-imbalance handling for the three modalities.

Imbalance this module exists to address (docs/1.docx, docs/MINDSCOPE_Blueprint.pdf
Sec. 02):
  - FER facial: Disgust 436 vs Happy 7215 (~16x minority class)
  - RAVDESS speech: Neutral 96 vs the other 7 classes at 192 each (no "strong"
    intensity recorded for neutral)
"""
from __future__ import annotations

import numpy as np


def facial_train_transform():
    """Albumentations pipeline for FER training: mild geometric/intensity jitter
    plus targeted augmentation for the minority Disgust class is applied by the
    caller via `oversample_minority_indices`, not baked into the transform itself
    (so the majority classes aren't over-augmented too).
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


def facial_eval_transform():
    import albumentations as A

    return A.Compose([])  # no augmentation at eval time


def oversample_minority_indices(labels: np.ndarray, target_count: int | None = None) -> np.ndarray:
    """Return an index array that oversamples minority classes up to
    `target_count` (defaults to the majority class count), for use with
    torch's SubsetRandomSampler / WeightedRandomSampler as a SMOTE-lite
    alternative for image data (SMOTE itself operates on feature vectors,
    not raw pixels -- see `tabular_smote` below for the FT-Transformer path).
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


def tabular_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """SMOTE oversampling for the tabular FT-Transformer path
    (imbalanced-learn), per docs/MINDSCOPE_Blueprint.pdf Sec. 04.
    """
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X, y)


def spec_augment(mel_spectrogram: np.ndarray, freq_mask_param: int = 15, time_mask_param: int = 25):
    """SpecAugment (Park et al. 2019) frequency + time masking for log-Mel
    spectrograms -- the augmentation for the CNN-BiLSTM speech fallback.
    `mel_spectrogram`: (n_mels, n_frames) array.
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
