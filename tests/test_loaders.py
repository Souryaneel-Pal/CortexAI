from src.data.loaders import FacialEmotionDataset, SpeechEmotionDataset, TabularMentalHealthDataset
from src.data.schemas import TABULAR_FEATURE_COLUMNS


def test_facial_dataset_loads_all_classes(synthetic_facial_dir):
    ds = FacialEmotionDataset(synthetic_facial_dir)
    assert len(ds) == 7 * 3  # 7 emotions x 3 synthetic images each
    image, label = ds[0]
    assert image.shape == (1, 48, 48)  # normalized (C, H, W) float tensor
    assert image.dtype.is_floating_point
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert 0 <= label <= 6


def test_speech_dataset_parses_labels_from_filename(synthetic_speech_dir):
    ds = SpeechEmotionDataset(synthetic_speech_dir, sample_rate=16000, max_duration_sec=4.0)
    assert len(ds) == 15  # 1 neutral (no strong intensity) + 7 * 2 others
    waveform, label, meta = ds[0]
    assert waveform.ndim == 1
    assert waveform.shape[0] == 16000 * 4  # pad/truncated to a fixed length for batching
    assert 0 <= label <= 7
    assert meta["gender"] in ("male", "female")


def test_speech_dataset_batches_via_default_collate(synthetic_speech_dir):
    from torch.utils.data import DataLoader

    ds = SpeechEmotionDataset(synthetic_speech_dir)
    loader = DataLoader(ds, batch_size=4)
    waveforms, labels, meta = next(iter(loader))
    assert waveforms.shape == (4, 16000 * 4)
    assert labels.shape == (4,)
    assert len(meta["gender"]) == 4


def test_facial_dataset_batches_via_default_collate(synthetic_facial_dir):
    from torch.utils.data import DataLoader

    ds = FacialEmotionDataset(synthetic_facial_dir)
    loader = DataLoader(ds, batch_size=5)
    images, labels = next(iter(loader))
    assert images.shape == (5, 1, 48, 48)
    assert labels.shape == (5,)


def test_tabular_dataset_loads_expected_shape(synthetic_tabular_csv):
    ds = TabularMentalHealthDataset(synthetic_tabular_csv)
    assert len(ds) == 40
    features, label, scores = ds[0]
    assert features.shape == (len(TABULAR_FEATURE_COLUMNS),)
    assert scores.shape == (3,)
    assert label in ("Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress")


def test_facial_dataset_missing_path_raises_clear_error(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="not been supplied yet"):
        FacialEmotionDataset(tmp_path / "does_not_exist")


def test_tabular_dataset_missing_path_raises_clear_error(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="not been supplied yet"):
        TabularMentalHealthDataset(tmp_path / "missing.csv")
