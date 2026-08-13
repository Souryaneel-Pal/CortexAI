from src.data.emotion_stress_map import (
    FACIAL_EMOTION_TO_STRESS,
    SPEECH_EMOTION_TO_STRESS,
    facial_logits_to_stress_distribution,
    speech_logits_to_stress_distribution,
)
from src.data.schemas import StressLevel


def test_facial_mapping_matches_docs_table3():
    # docs/1.docx Table 3
    assert FACIAL_EMOTION_TO_STRESS["Happy"] == StressLevel.HEALTHY
    assert FACIAL_EMOTION_TO_STRESS["Neutral"] == StressLevel.HEALTHY
    assert FACIAL_EMOTION_TO_STRESS["Sad"] == StressLevel.MILD_STRESS
    assert FACIAL_EMOTION_TO_STRESS["Surprise"] == StressLevel.MILD_STRESS
    assert FACIAL_EMOTION_TO_STRESS["Fear"] == StressLevel.MODERATE_STRESS
    assert FACIAL_EMOTION_TO_STRESS["Disgust"] == StressLevel.MODERATE_STRESS
    assert FACIAL_EMOTION_TO_STRESS["Angry"] == StressLevel.SEVERE_STRESS


def test_speech_mapping_matches_docs_table1():
    # docs/1.docx Table 1
    assert SPEECH_EMOTION_TO_STRESS["neutral"] == StressLevel.HEALTHY
    assert SPEECH_EMOTION_TO_STRESS["calm"] == StressLevel.HEALTHY
    assert SPEECH_EMOTION_TO_STRESS["happy"] == StressLevel.HEALTHY
    assert SPEECH_EMOTION_TO_STRESS["sad"] == StressLevel.MILD_STRESS
    assert SPEECH_EMOTION_TO_STRESS["surprised"] == StressLevel.MILD_STRESS
    assert SPEECH_EMOTION_TO_STRESS["fearful"] == StressLevel.MODERATE_STRESS
    assert SPEECH_EMOTION_TO_STRESS["angry"] == StressLevel.MODERATE_STRESS
    assert SPEECH_EMOTION_TO_STRESS["disgust"] == StressLevel.SEVERE_STRESS


def test_mappings_deliberately_disagree_on_disgust_and_angry():
    # The whole point: face Disgust=Moderate/Angry=Severe, speech is reversed.
    assert FACIAL_EMOTION_TO_STRESS["Disgust"] != SPEECH_EMOTION_TO_STRESS["disgust"]
    assert FACIAL_EMOTION_TO_STRESS["Angry"] != SPEECH_EMOTION_TO_STRESS["angry"]


def test_facial_distribution_projection_sums_to_one():
    probs = {"Happy": 0.5, "Neutral": 0.2, "Sad": 0.1, "Angry": 0.2}
    dist = facial_logits_to_stress_distribution(probs)
    assert len(dist) == 4
    assert round(sum(dist), 6) == 1.0
    assert round(dist[StressLevel.HEALTHY], 6) == 0.7  # Happy + Neutral
    assert round(dist[StressLevel.SEVERE_STRESS], 6) == 0.2  # Angry


def test_speech_distribution_projection_sums_to_one():
    probs = {"neutral": 0.3, "disgust": 0.3, "fearful": 0.4}
    dist = speech_logits_to_stress_distribution(probs)
    assert round(sum(dist), 6) == 1.0
    assert round(dist[StressLevel.SEVERE_STRESS], 6) == 0.3  # disgust
    assert round(dist[StressLevel.MODERATE_STRESS], 6) == 0.4  # fearful
