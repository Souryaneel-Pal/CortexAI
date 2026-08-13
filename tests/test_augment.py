import numpy as np

from src.data.augment import class_balanced_sample_weights, oversample_minority_indices, spec_augment, tabular_smote


def test_oversample_minority_indices_balances_counts():
    labels = np.array([0] * 20 + [1] * 5)
    idx = oversample_minority_indices(labels)
    resampled_labels = labels[idx]
    counts = np.bincount(resampled_labels)
    assert counts[0] == counts[1] == 20


def test_class_balanced_sample_weights_favor_minority():
    labels = np.array([0] * 100 + [1] * 5)
    weights = class_balanced_sample_weights(labels)
    minority_weight = weights[labels == 1][0]
    majority_weight = weights[labels == 0][0]
    assert minority_weight > majority_weight


def test_tabular_smote_balances_classes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 18))
    y = np.array([0] * 100 + [1] * 20)
    X_res, y_res = tabular_smote(X, y)
    counts = np.bincount(y_res)
    assert counts[0] == counts[1]


def test_spec_augment_preserves_shape():
    spec = np.random.rand(64, 100).astype(np.float32)
    augmented = spec_augment(spec)
    assert augmented.shape == spec.shape


# ---------------------------------------------------------------------------
# Train-vs-eval separation. Augmenting the validation split silently inflates
# every metric, so these pin that augmentation is strictly training-only.
# ---------------------------------------------------------------------------
def test_facial_eval_mode_returns_untouched_pixels(synthetic_facial_dir):
    """Two reads of the same eval-mode sample must be byte-identical, and must
    equal the raw file on disk."""
    from PIL import Image

    from src.data.loaders import FacialEmotionDataset

    ds = FacialEmotionDataset(synthetic_facial_dir, train=False)
    first, _ = ds[0]
    second, _ = ds[0]
    assert np.array_equal(first.numpy(), second.numpy())

    source_path, _label = ds.samples[0]
    on_disk = np.array(Image.open(source_path).convert("L"), dtype=np.float32) / 255.0
    assert np.allclose(first.numpy()[0], on_disk)


def test_facial_train_mode_actually_perturbs_pixels(synthetic_facial_dir):
    """Training mode must produce varying views of the same index."""
    from src.data.loaders import FacialEmotionDataset

    ds = FacialEmotionDataset(synthetic_facial_dir, train=True)
    views = [ds[0][0].numpy() for _ in range(12)]
    assert any(not np.array_equal(views[0], v) for v in views[1:]), (
        "train=True produced 12 identical views -- augmentation is not being applied"
    )


def test_speech_eval_mode_returns_untouched_waveform(synthetic_speech_dir):
    from src.data.loaders import SpeechEmotionDataset

    ds = SpeechEmotionDataset(synthetic_speech_dir, train=False)
    first, _label, _meta = ds[0]
    second, _label2, _meta2 = ds[0]
    assert np.array_equal(first.numpy(), second.numpy())


def test_speech_train_mode_actually_perturbs_waveform(synthetic_speech_dir):
    from src.data.loaders import SpeechEmotionDataset

    ds = SpeechEmotionDataset(synthetic_speech_dir, train=True)
    views = [ds[0][0].numpy() for _ in range(8)]
    assert any(not np.array_equal(views[0], v) for v in views[1:]), (
        "train=True produced identical waveforms -- augmentation is not being applied"
    )


def test_spec_augment_masks_only_in_training_mode():
    """SpecAugment lives inside CNNBiLSTMEncoder.forward and must be gated on
    `self.training`, so eval-mode inference sees unmasked spectrograms."""
    import torch

    from src.models.speech_net import CNNBiLSTMEncoder

    torch.manual_seed(0)
    model = CNNBiLSTMEncoder(spec_augment=True)
    waveform = torch.randn(2, 16000)

    model.eval()
    with torch.no_grad():
        a, _ = model(waveform)
        b, _ = model(waveform)
    assert torch.allclose(a, b), "eval-mode forward is not deterministic -- SpecAugment leaked into eval"


def test_facial_minority_class_uses_stronger_pipeline(synthetic_facial_dir):
    """Disgust (the 16.5x minority) must get the targeted pipeline, not the
    mild majority one."""
    from src.data.augment import FACIAL_MINORITY_CLASSES
    from src.data.loaders import FACIAL_EMOTION_NAME_TO_IDX, FacialEmotionDataset

    ds = FacialEmotionDataset(synthetic_facial_dir, train=True)
    minority_idx = FACIAL_EMOTION_NAME_TO_IDX[FACIAL_MINORITY_CLASSES[0]]
    majority_idx = FACIAL_EMOTION_NAME_TO_IDX["Happy"]

    assert ds._transform_for(minority_idx) is ds._minority_transform
    assert ds._transform_for(majority_idx) is ds._majority_transform


def test_smote_carries_regression_targets_with_synthetic_rows():
    """SMOTE synthesises minority feature rows; the 3 score targets must be
    interpolated alongside them, or synthetic rows get mismatched labels."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 18))
    scores = rng.uniform(0, 30, size=(120, 3))
    y = np.array([0] * 100 + [1] * 20)

    joint = np.hstack([X, scores])
    joint_res, y_res = tabular_smote(joint, y)
    X_res, scores_res = joint_res[:, :18], joint_res[:, 18:]

    assert len(X_res) == len(scores_res) == len(y_res)
    assert np.bincount(y_res)[0] == np.bincount(y_res)[1]
    # Interpolated scores stay inside the observed range (SMOTE is convex).
    assert scores_res.min() >= scores.min() - 1e-6
    assert scores_res.max() <= scores.max() + 1e-6
