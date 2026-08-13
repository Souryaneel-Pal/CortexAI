import pandas as pd
import pytest

from src.explain.masked_distress import (
    PhysiologicalReferenceStats,
    masked_distress_index,
    physiological_arousal_score,
)


@pytest.fixture
def reference():
    df = pd.DataFrame(
        {
            "Heart_Rate_BPM": [70, 72, 68, 75, 71],
            "HRV_Index": [50, 48, 52, 45, 49],
            "Skin_Temperature": [33.0, 33.2, 32.9, 33.1, 33.0],
            "GSR_Level": [2.0, 2.1, 1.9, 2.2, 2.0],
        }
    )
    return PhysiologicalReferenceStats.fit_from_dataframe(df)


def test_physiological_arousal_score_average_row_near_half(reference):
    average_row = {"Heart_Rate_BPM": 71.2, "HRV_Index": 48.8, "Skin_Temperature": 33.04, "GSR_Level": 2.04}
    score = physiological_arousal_score(average_row, reference)
    assert abs(score - 0.5) < 0.05


def test_physiological_arousal_score_high_stress_row_is_high(reference):
    # Elevated HR/GSR, low HRV, low skin temp -- all point toward high arousal.
    high_arousal_row = {"Heart_Rate_BPM": 95, "HRV_Index": 30, "Skin_Temperature": 31.5, "GSR_Level": 3.5}
    score = physiological_arousal_score(high_arousal_row, reference)
    assert score > 0.7


def test_mdi_high_when_face_calm_but_voice_and_physio_high_arousal(reference):
    facial_probs = {"Happy": 0.9, "Neutral": 0.1}
    speech_probs = {"disgust": 0.9, "neutral": 0.1}  # disgust -> Severe (docs/1.docx Table 1)
    high_arousal_row = {"Heart_Rate_BPM": 95, "HRV_Index": 30, "Skin_Temperature": 31.5, "GSR_Level": 3.5}

    result = masked_distress_index(facial_probs, speech_probs, high_arousal_row, reference)
    assert result["mdi"] > 0.7
    assert result["flag"] is True
    assert result["dominant_contradiction"] in ("voice", "physiology")


def test_mdi_low_when_face_already_reads_distressed(reference):
    facial_probs = {"Angry": 0.9, "Neutral": 0.1}  # Angry -> Severe (docs/1.docx Table 3)
    speech_probs = {"disgust": 0.9, "neutral": 0.1}
    high_arousal_row = {"Heart_Rate_BPM": 95, "HRV_Index": 30, "Skin_Temperature": 31.5, "GSR_Level": 3.5}

    result = masked_distress_index(facial_probs, speech_probs, high_arousal_row, reference)
    assert result["mdi"] < 0.2  # no contradiction: face already reads high-arousal
    assert result["flag"] is False


def test_mdi_low_when_everything_calm(reference):
    facial_probs = {"Happy": 1.0}
    speech_probs = {"calm": 1.0}
    # Genuinely low-arousal physiology (below-average HR/GSR, above-average
    # HRV/skin temp), not merely "average" -- an average row scores ~0.5
    # arousal by construction (sigmoid of a zero z-score), which is not "calm".
    low_arousal_row = {"Heart_Rate_BPM": 55, "HRV_Index": 70, "Skin_Temperature": 34.0, "GSR_Level": 1.2}
    result = masked_distress_index(facial_probs, speech_probs, low_arousal_row, reference)
    assert result["mdi"] < 0.3
