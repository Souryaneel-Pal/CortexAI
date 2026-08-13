from src.data.loaders import (
    FACIAL_EMOTION_NAME_TO_IDX,
    FacialEmotionDataset,
    SpeechEmotionDataset,
    TabularMentalHealthDataset,
)
from src.data.schemas import STRESS_LABEL_NAME_TO_LEVEL, StressLevel, TABULAR_FEATURE_COLUMNS


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
    # Labels collate straight into a loss function, so they are integer
    # StressLevel indices; the raw strings stay on `.raw_labels` / `.df`.
    assert int(label) in {int(level) for level in StressLevel}
    assert ds.raw_labels[0] in ("Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress")


def test_tabular_dataset_labels_align_with_raw_strings(synthetic_tabular_csv):
    ds = TabularMentalHealthDataset(synthetic_tabular_csv)
    for encoded, raw in zip(ds.labels, ds.raw_labels):
        assert int(encoded) == int(STRESS_LABEL_NAME_TO_LEVEL[raw])


def test_tabular_dataset_indices_subset_selects_rows(synthetic_tabular_csv):
    import numpy as np

    full = TabularMentalHealthDataset(synthetic_tabular_csv)
    subset = TabularMentalHealthDataset(synthetic_tabular_csv, indices=np.array([2, 5, 7]))
    assert len(subset) == 3
    assert np.allclose(subset.features[0], full.features[2])
    assert int(subset.labels[1]) == int(full.labels[5])


def test_facial_dataset_missing_path_raises_clear_error(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="Dataset Access"):
        FacialEmotionDataset(tmp_path / "does_not_exist")


def test_tabular_dataset_missing_path_raises_clear_error(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="Dataset Access"):
        TabularMentalHealthDataset(tmp_path / "missing.csv")


def test_speech_dataset_deduplicates_repeated_clips(synthetic_speech_dir, tmp_path):
    """The shipped archive nests a byte-identical copy of every Actor_XX
    folder under `audio_speech_actors_01-24/`; the loader must not count it
    twice (2,880 files on disk, 1,440 unique clips)."""
    import shutil

    root = tmp_path / "Audios"
    shutil.copytree(synthetic_speech_dir, root / "Actor_01")
    shutil.copytree(synthetic_speech_dir, root / "audio_speech_actors_01-24" / "Actor_01")

    ds = SpeechEmotionDataset(root)
    assert len(ds) == 15  # not 30 -- the duplicate copy is dropped by filename


# ---------------------------------------------------------------------------
# FER+ labels
#
# FER+ re-annotates the same FER2013 images and is joined purely on the row
# index encoded in each filename. That join is silent when it breaks -- labels
# would just be wrong -- so it gets pinned.
# ---------------------------------------------------------------------------
def test_ferplus_majority_vote_maps_to_the_seven_class_schema(tmp_path):
    import pandas as pd

    from src.data.loaders import FERPLUS_VOTE_COLUMNS, load_ferplus_labels

    rows = [
        # index 0: clear happiness
        ("Training", "fer0.png", 0, 9, 0, 1, 0, 0, 0, 0, 0, 0),
        # index 1: clear anger
        ("Training", "fer1.png", 1, 0, 0, 0, 8, 1, 0, 0, 0, 0),
        # index 2: contempt wins -- excluded, no slot in the 7-class schema
        ("Training", "fer2.png", 1, 0, 0, 0, 0, 0, 0, 9, 0, 0),
        # index 3: "not a face" wins -- excluded
        ("Training", "fer3.png", 0, 0, 0, 0, 0, 0, 0, 0, 1, 9),
        # A non-Training row must never enter the index space.
        ("PublicTest", "fer4.png", 10, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ]
    csv = tmp_path / "fer2013new.csv"
    pd.DataFrame(rows, columns=["Usage", "Image name", *FERPLUS_VOTE_COLUMNS]).to_csv(csv, index=False)

    labels = load_ferplus_labels(csv)

    assert set(labels) == {0, 1}, "contempt / NF rows must be excluded, PublicTest rows must not appear"
    assert labels[0] == FACIAL_EMOTION_NAME_TO_IDX["Happy"]
    assert labels[1] == FACIAL_EMOTION_NAME_TO_IDX["Angry"]


def test_ferplus_indices_are_relative_to_the_training_split(tmp_path):
    """FER+ covers all 35,887 FER2013 rows but the extracted folders hold only
    the 28,709 Training ones. If PublicTest rows were kept, every index after
    the first would point at the wrong image."""
    import pandas as pd

    from src.data.loaders import FERPLUS_VOTE_COLUMNS, load_ferplus_labels

    rows = [
        ("PublicTest", "a.png", 10, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ("Training", "b.png", 0, 10, 0, 0, 0, 0, 0, 0, 0, 0),
    ]
    csv = tmp_path / "fer2013new.csv"
    pd.DataFrame(rows, columns=["Usage", "Image name", *FERPLUS_VOTE_COLUMNS]).to_csv(csv, index=False)

    labels = load_ferplus_labels(csv)
    # The single Training row must be index 0, not index 1.
    assert labels == {0: FACIAL_EMOTION_NAME_TO_IDX["Happy"]}


def test_ferplus_min_vote_fraction_drops_ambiguous_images(tmp_path):
    import pandas as pd

    from src.data.loaders import FERPLUS_VOTE_COLUMNS, load_ferplus_labels

    rows = [
        ("Training", "a.png", 5, 5, 0, 0, 0, 0, 0, 0, 0, 0),  # 50% agreement
        ("Training", "b.png", 9, 1, 0, 0, 0, 0, 0, 0, 0, 0),  # 90% agreement
    ]
    csv = tmp_path / "fer2013new.csv"
    pd.DataFrame(rows, columns=["Usage", "Image name", *FERPLUS_VOTE_COLUMNS]).to_csv(csv, index=False)

    assert len(load_ferplus_labels(csv, min_vote_fraction=0.0)) == 2
    assert set(load_ferplus_labels(csv, min_vote_fraction=0.8)) == {1}


def test_ferplus_on_non_indexed_filenames_fails_loudly(synthetic_facial_dir, tmp_path):
    """FER+ joins on the row index in the filename. A corpus whose files are
    not index-named cannot be matched, and that must be an explicit error --
    not a silent empty dataset, and not the generic "no images found" message,
    which would send someone to re-download data they already have."""
    import pandas as pd
    import pytest

    from src.data.loaders import FERPLUS_VOTE_COLUMNS

    csv = tmp_path / "fer2013new.csv"
    pd.DataFrame(
        [("Training", "fer0.png", 0, 10, 0, 0, 0, 0, 0, 0, 0, 0)],
        columns=["Usage", "Image name", *FERPLUS_VOTE_COLUMNS],
    ).to_csv(csv, index=False)

    with pytest.raises(ValueError, match="FER2013 row index"):
        FacialEmotionDataset(synthetic_facial_dir, label_source="ferplus", ferplus_csv=csv)
