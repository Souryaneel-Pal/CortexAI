from src.data.loaders import FacialEmotionDataset, SpeechEmotionDataset, TabularMentalHealthDataset
from src.data.schemas import TABULAR_FEATURE_COLUMNS


def test_facial_dataset_loads_all_classes(synthetic_facial_dir):
    ds = FacialEmotionDataset(synthetic_facial_dir)
    assert len(ds) == 7 * 3  # 7 emotions x 3 synthetic images each
    image, label = ds[0]
    assert image.shape == (48, 48)
    assert 0 <= label <= 6


def test_speech_dataset_parses_labels_from_filename(synthetic_speech_dir):
    ds = SpeechEmotionDataset(synthetic_speech_dir)
    assert len(ds) == 15  # 1 neutral (no strong intensity) + 7 * 2 others
    waveform, label, meta = ds[0]
    assert waveform.ndim == 1
    assert 0 <= label <= 7
    assert meta["gender"] in ("male", "female")


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
