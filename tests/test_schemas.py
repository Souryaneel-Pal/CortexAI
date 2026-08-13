from src.data.schemas import (
    FACIAL_CLASS_COUNTS,
    RAVDESS_CLASS_COUNTS,
    TABULAR_FEATURE_COLUMNS,
    ravdess_actor_gender,
    parse_ravdess_filename,
)


def test_18_tabular_features_exactly():
    assert len(TABULAR_FEATURE_COLUMNS) == 18
    assert len(set(TABULAR_FEATURE_COLUMNS)) == 18  # no duplicates


def test_facial_class_counts_match_docs():
    assert FACIAL_CLASS_COUNTS["Happy"] == 7215
    assert FACIAL_CLASS_COUNTS["Disgust"] == 436
    assert sum(FACIAL_CLASS_COUNTS.values()) == 28709


def test_ravdess_class_counts_match_docs():
    assert RAVDESS_CLASS_COUNTS["neutral"] == 96
    assert RAVDESS_CLASS_COUNTS["calm"] == 192
    assert sum(RAVDESS_CLASS_COUNTS.values()) == 1440


def test_ravdess_actor_gender():
    assert ravdess_actor_gender(1) == "male"
    assert ravdess_actor_gender(12) == "female"
    assert ravdess_actor_gender(24) == "female"
    assert ravdess_actor_gender(23) == "male"


def test_parse_ravdess_filename_example_from_docs():
    # docs/1.docx worked example: 03-01-06-01-02-01-12.wav
    meta = parse_ravdess_filename("03-01-06-01-02-01-12.wav")
    assert meta["modality"] == "03"  # audio-only
    assert meta["vocal_channel"] == "01"  # speech
    assert meta["emotion_label"] == "fearful"
    assert meta["intensity"] == "01"  # normal
    assert meta["statement"] == "02"  # "Dogs are sitting by the door"
    assert meta["repetition"] == "01"
    assert meta["actor_id"] == 12
    assert meta["gender"] == "female"


def test_parse_ravdess_filename_rejects_malformed():
    import pytest

    with pytest.raises(ValueError):
        parse_ravdess_filename("not-a-valid-name.wav")
