"""Canonical schema for CortexAI's three data modalities.

Column names, class labels, and score ranges are taken verbatim from
docs/1.docx ("Dataset URL" / "Dataset Description" section). Nothing here is
inferred — every name below appears in that document.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


# --------------------------------------------------------------------------
# Shared severity axis (docs/Problem_Statement.docx, docs/1.docx)
# --------------------------------------------------------------------------
class StressLevel(IntEnum):
    HEALTHY = 0
    MILD_STRESS = 1
    MODERATE_STRESS = 2
    SEVERE_STRESS = 3


STRESS_LEVEL_NAMES = {
    StressLevel.HEALTHY: "Healthy",
    StressLevel.MILD_STRESS: "Mild_Stress",
    StressLevel.MODERATE_STRESS: "Moderate_Stress",
    StressLevel.SEVERE_STRESS: "Severe_Stress",
}
STRESS_LEVEL_DISPLAY = {
    StressLevel.HEALTHY: "Healthy",
    StressLevel.MILD_STRESS: "Mild Stress",
    StressLevel.MODERATE_STRESS: "Moderate Stress",
    StressLevel.SEVERE_STRESS: "Severe Stress",
}

# Inverse of STRESS_LEVEL_NAMES -- maps the raw Mental_Health_Status string in
# numerical.csv back onto the StressLevel enum. The single source of truth
# for this mapping; import it rather than redefining it locally.
STRESS_LABEL_NAME_TO_LEVEL = {name: level for level, name in STRESS_LEVEL_NAMES.items()}

# --------------------------------------------------------------------------
# Facial modality — FER-2013-style, 48x48 grayscale, 7 emotions
# --------------------------------------------------------------------------
FACIAL_EMOTIONS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}
FACIAL_IMAGE_SIZE = 48  # pixels, grayscale (1 channel)

# Class counts as documented (docs/1.docx, Table 2) — used to size class-balanced
# focal loss / SMOTE, and as the assertion validate_datasets.py checks against.
FACIAL_CLASS_COUNTS = {
    "Angry": 3995,
    "Disgust": 436,
    "Fear": 4097,
    "Happy": 7215,
    "Neutral": 4965,
    "Sad": 4830,
    "Surprise": 3171,
}

# --------------------------------------------------------------------------
# Speech modality — RAVDESS, 1440 .wav files, 8 emotions, gender-balanced
# --------------------------------------------------------------------------
SPEECH_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# RAVDESS filename convention (docs/1.docx):
# modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
RAVDESS_FIELDS = (
    "modality",
    "vocal_channel",
    "emotion",
    "intensity",
    "statement",
    "repetition",
    "actor",
)
RAVDESS_MODALITY = {"01": "full-AV", "02": "video-only", "03": "audio-only"}
RAVDESS_VOCAL_CHANNEL = {"01": "speech", "02": "song"}
RAVDESS_INTENSITY = {"01": "normal", "02": "strong"}
RAVDESS_STATEMENT = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}

RAVDESS_CLASS_COUNTS = {
    "neutral": 96,
    "calm": 192,
    "happy": 192,
    "sad": 192,
    "angry": 192,
    "fearful": 192,
    "disgust": 192,
    "surprised": 192,
}


def ravdess_actor_gender(actor_id: int) -> str:
    """Odd actor IDs are male, even are female (docs/1.docx)."""
    return "male" if actor_id % 2 == 1 else "female"


def parse_ravdess_filename(filename: str) -> dict:
    """Parse a RAVDESS filename like '03-01-06-01-02-01-12.wav' into its 7 fields."""
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(
            f"Expected 7 hyphen-separated fields in RAVDESS filename, got {len(parts)}: {filename!r}"
        )
    fields = dict(zip(RAVDESS_FIELDS, parts))
    actor_id = int(fields["actor"])
    return {
        **fields,
        "actor_id": actor_id,
        "gender": ravdess_actor_gender(actor_id),
        "emotion_code": fields["emotion"],
        "emotion_label": SPEECH_EMOTIONS[fields["emotion"]],
    }


# --------------------------------------------------------------------------
# Tabular modality — 4000 rows, 18 features, the only labelled ground truth
# --------------------------------------------------------------------------
TABULAR_FEATURES_BY_DOMAIN = {
    "behavioural": [
        "Sleep_Quality",
        "Social_Engagement",
        "Daily_App_Usage_Min",
        "Typing_Speed_WPM",
        "Session_Frequency",
        "Idle_Time_Min",
    ],
    "facial": [
        "Facial_Emotion_Variance",
        "Eye_Blink_Rate",
        "Smile_Intensity",
        "Head_Motion_Index",
    ],
    "acoustic": [
        "MFCC_Mean",
        "MFCC_Variance",
        "Pitch_Mean",
        "Speech_Rate",
    ],
    "physiological": [
        "Heart_Rate_BPM",
        "HRV_Index",
        "Skin_Temperature",
        "GSR_Level",
    ],
}

TABULAR_FEATURE_COLUMNS: list[str] = [
    col for cols in TABULAR_FEATURES_BY_DOMAIN.values() for col in cols
]
assert len(TABULAR_FEATURE_COLUMNS) == 18, "docs/1.docx specifies exactly 18 features"

TABULAR_TARGET_CLASS_COLUMN = "Mental_Health_Status"
TABULAR_TARGET_SCORE_COLUMNS = ["Depression_Score", "Anxiety_Score", "Stress_Score"]

# Score ranges (docs/1.docx) — used for regression-head output clamping/scaling and
# for the 0-1 normalisation used by the consistency loss term (src/train/losses.py).
SCORE_RANGES = {
    "Depression_Score": (0, 34),
    "Anxiety_Score": (0, 24),
    "Stress_Score": (0, 39),
}

TABULAR_ROW_COUNT = 4000
FACIAL_IMAGE_COUNT = 28709
SPEECH_CLIP_COUNT = 1440


@dataclass(frozen=True)
class ModalityEmbedding:
    """Contract every per-modality encoder outputs, consumed by src/models/fusion.py."""

    EMBED_DIM = 256
