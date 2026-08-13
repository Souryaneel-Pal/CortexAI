from src.data.loaders import FacialEmotionDataset, FusionPairDataset, SpeechEmotionDataset, TabularMentalHealthDataset
from src.data.emotion_stress_map import FACIAL_EMOTION_TO_STRESS, SPEECH_EMOTION_TO_STRESS
from src.data.schemas import FACIAL_EMOTIONS, StressLevel


def test_fusion_pair_dataset_matches_stress_tiers(
    synthetic_tabular_csv, synthetic_facial_dir, synthetic_speech_dir
):
    tabular_ds = TabularMentalHealthDataset(synthetic_tabular_csv)
    facial_ds = FacialEmotionDataset(synthetic_facial_dir)
    speech_ds = SpeechEmotionDataset(synthetic_speech_dir)

    pair_ds = FusionPairDataset(tabular_ds, facial_ds, speech_ds, seed=0)
    assert len(pair_ds) == len(tabular_ds)

    idx_to_facial_name = {v: k for k, v in {name: idx for idx, name in FACIAL_EMOTIONS.items()}.items()}
    stress_label_to_tier = {
        "Healthy": StressLevel.HEALTHY,
        "Mild_Stress": StressLevel.MILD_STRESS,
        "Moderate_Stress": StressLevel.MODERATE_STRESS,
        "Severe_Stress": StressLevel.SEVERE_STRESS,
    }

    for i in range(len(pair_ds)):
        features, label, scores, image, waveform = pair_ds[i]
        row_tier = stress_label_to_tier[label]

        facial_idx = pair_ds._facial_pairs[i]
        _source, facial_label_idx = facial_ds.samples[facial_idx]
        facial_emotion_name = idx_to_facial_name[facial_label_idx]
        assert FACIAL_EMOTION_TO_STRESS[facial_emotion_name] == row_tier

        speech_idx = pair_ds._speech_pairs[i]
        _path, meta = speech_ds.samples[speech_idx]
        assert SPEECH_EMOTION_TO_STRESS[meta["emotion_label"]] == row_tier

        assert image.shape == (1, 48, 48)
        assert waveform.ndim == 1


def test_fusion_pair_dataset_resample_changes_pairing(
    synthetic_tabular_csv, synthetic_facial_dir, synthetic_speech_dir
):
    tabular_ds = TabularMentalHealthDataset(synthetic_tabular_csv)
    facial_ds = FacialEmotionDataset(synthetic_facial_dir)
    speech_ds = SpeechEmotionDataset(synthetic_speech_dir)

    pair_ds = FusionPairDataset(tabular_ds, facial_ds, speech_ds, seed=0)
    before = pair_ds._facial_pairs.copy()
    pair_ds.resample()
    after = pair_ds._facial_pairs
    assert not (before == after).all()  # extremely unlikely to be identical by chance


def test_fusion_pair_dataset_batches_via_default_collate(
    synthetic_tabular_csv, synthetic_facial_dir, synthetic_speech_dir
):
    from torch.utils.data import DataLoader

    tabular_ds = TabularMentalHealthDataset(synthetic_tabular_csv)
    facial_ds = FacialEmotionDataset(synthetic_facial_dir)
    speech_ds = SpeechEmotionDataset(synthetic_speech_dir)
    pair_ds = FusionPairDataset(tabular_ds, facial_ds, speech_ds, seed=0)

    loader = DataLoader(pair_ds, batch_size=8)
    features, labels, scores, images, waveforms = next(iter(loader))
    assert features.shape[1] == 18
    assert images.shape[1:] == (1, 48, 48)
    assert waveforms.ndim == 2
