"""The Masked-Distress Index (MDI) -- CortexAI's signature explainability
feature (docs/PROJECT_PLAN.md P3, docs/MINDSCOPE_Blueprint.pdf Sec. 06):

    "A novel composite ... when the face reads calm/happy but voice and
    physiology read high-arousal, the modalities contradict. MDI quantifies
    that gap. Clinically, a high MDI flags masked or suppressed distress --
    precisely the presentation that self-report questionnaires miss."

Design (documented here since this formula is CortexAI's own construction,
not from the source docs -- it must be validated against real clinical data
before any real-world use, per the Responsible-AI framing):

    face_calm    = P(face stress-tier in {Healthy, Mild})   -- from
                   src/data/emotion_stress_map.facial_logits_to_stress_distribution
    voice_high   = P(speech stress-tier in {Moderate, Severe}) -- from
                   src/data/emotion_stress_map.speech_logits_to_stress_distribution
    physio_high  = physiological_arousal_score(...) in [0, 1] -- a z-scored
                   composite of Heart_Rate_BPM, HRV_Index, Skin_Temperature,
                   GSR_Level (the 4 physiological tabular columns)

    MDI = face_calm * max(voice_high, physio_high)   in [0, 1]

MDI is high only when the face specifically reads calm/healthy AND at least
one other modality reads high-arousal -- a face that already reads distressed
produces a low MDI (no contradiction to flag), by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.emotion_stress_map import facial_logits_to_stress_distribution, speech_logits_to_stress_distribution
from src.data.schemas import StressLevel

# Documented, clinically-unvalidated sign assumptions for each physiological
# feature's relationship to sympathetic arousal (placeholder heuristic --
# flagged in PROJECT_PLAN.md P4 alongside the rest of the placeholder
# clinical content pending real-world validation):
#   Heart_Rate_BPM:    higher  -> higher arousal
#   HRV_Index:         lower   -> higher arousal (low HRV = high stress)
#   Skin_Temperature:  lower   -> higher arousal (peripheral vasoconstriction)
#   GSR_Level:         higher  -> higher arousal (sympathetic skin response)
PHYSIOLOGICAL_AROUSAL_DIRECTION = {
    "Heart_Rate_BPM": 1,
    "HRV_Index": -1,
    "Skin_Temperature": -1,
    "GSR_Level": 1,
}

DEFAULT_MDI_FLAG_THRESHOLD = 0.5


@dataclass(frozen=True)
class PhysiologicalReferenceStats:
    """Population mean/std for z-scoring the 4 physiological features, fit
    once from the real tabular dataset (see `fit_from_dataframe`). Required
    because raw values (e.g. GSR_Level's units) aren't meaningful without a
    population reference.
    """

    mean: dict[str, float]
    std: dict[str, float]

    @classmethod
    def fit_from_dataframe(cls, df) -> "PhysiologicalReferenceStats":
        columns = list(PHYSIOLOGICAL_AROUSAL_DIRECTION.keys())
        return cls(mean=df[columns].mean().to_dict(), std=df[columns].std().to_dict())


def physiological_arousal_score(row: dict[str, float], reference: PhysiologicalReferenceStats) -> float:
    """row: dict with the 4 physiological feature values for one participant.
    Returns a [0, 1] arousal score: 0.5 = population average, higher = more
    sympathetic-arousal indicators than average, via a sigmoid over the
    mean z-score across the 4 (direction-corrected) features.
    """
    z_scores = []
    for feature, direction in PHYSIOLOGICAL_AROUSAL_DIRECTION.items():
        std = reference.std[feature] or 1.0  # guard against zero-variance columns
        z = (row[feature] - reference.mean[feature]) / std
        z_scores.append(direction * z)
    mean_z = float(np.mean(z_scores))
    return float(1.0 / (1.0 + np.exp(-mean_z)))  # sigmoid squashes to [0, 1]


def masked_distress_index(
    facial_emotion_probs: dict[str, float],
    speech_emotion_probs: dict[str, float],
    physiological_row: dict[str, float],
    reference: PhysiologicalReferenceStats,
    threshold: float = DEFAULT_MDI_FLAG_THRESHOLD,
) -> dict:
    """Returns {"mdi": float in [0,1], "face_calm": float, "voice_high_arousal":
    float, "physio_high_arousal": float, "flag": bool, "dominant_contradiction":
    "voice" | "physiology" | None} -- the full breakdown, not just the scalar,
    so the UI can show *why* MDI fired (Objective 3's actionable-explanation
    requirement).
    """
    face_dist = facial_logits_to_stress_distribution(facial_emotion_probs)
    voice_dist = speech_logits_to_stress_distribution(speech_emotion_probs)

    face_calm = face_dist[int(StressLevel.HEALTHY)] + face_dist[int(StressLevel.MILD_STRESS)]
    voice_high_arousal = voice_dist[int(StressLevel.MODERATE_STRESS)] + voice_dist[int(StressLevel.SEVERE_STRESS)]
    physio_high_arousal = physiological_arousal_score(physiological_row, reference)

    mdi = face_calm * max(voice_high_arousal, physio_high_arousal)
    dominant = None
    if mdi > 0:
        dominant = "voice" if voice_high_arousal >= physio_high_arousal else "physiology"

    return {
        "mdi": mdi,
        "face_calm": face_calm,
        "voice_high_arousal": voice_high_arousal,
        "physio_high_arousal": physio_high_arousal,
        "flag": mdi >= threshold,
        "dominant_contradiction": dominant,
    }
