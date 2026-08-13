"""Emotion -> stress-severity bridge (docs/1.docx, docs/Proposal.pdf Sec. 03).

The facial and speech emotion vocabularies map onto the shared 4-tier severity
axis (Healthy -> Mild -> Moderate -> Severe) via two DIFFERENT tables. The
docs are explicit that these disagree (Disgust = Moderate for faces but
Severe for speech) and that this should NOT be forced into one rule:

    "Note the mapping differs slightly between the facial and speech tables
    -- do not force them into one rule, treat them as modality-specific
    priors reconciled by the fusion layer" (task brief, cross-checked
    against docs/1.docx Tables 1 & 3 and docs/Proposal.pdf Sec. 03).

So this module exposes two independent priors, `FACIAL_EMOTION_TO_STRESS`
and `SPEECH_EMOTION_TO_STRESS`, and deliberately does NOT expose a merged
"emotion -> stress" dict. Reconciliation across modalities happens only in
the fusion layer (src/models/fusion.py), which sees both as soft evidence
alongside the tabular ground truth, never as a hard label.
"""
from __future__ import annotations

from src.data.schemas import StressLevel

# --------------------------------------------------------------------------
# Facial emotion -> stress level (docs/1.docx, Table 3)
# --------------------------------------------------------------------------
FACIAL_EMOTION_TO_STRESS: dict[str, StressLevel] = {
    "Happy": StressLevel.HEALTHY,
    "Neutral": StressLevel.HEALTHY,
    "Sad": StressLevel.MILD_STRESS,
    "Surprise": StressLevel.MILD_STRESS,
    "Fear": StressLevel.MODERATE_STRESS,
    "Disgust": StressLevel.MODERATE_STRESS,
    "Angry": StressLevel.SEVERE_STRESS,
}

# --------------------------------------------------------------------------
# Speech emotion -> stress level (docs/1.docx, Table 1)
# --------------------------------------------------------------------------
SPEECH_EMOTION_TO_STRESS: dict[str, StressLevel] = {
    "neutral": StressLevel.HEALTHY,
    "calm": StressLevel.HEALTHY,
    "happy": StressLevel.HEALTHY,
    "sad": StressLevel.MILD_STRESS,
    "surprised": StressLevel.MILD_STRESS,
    "fearful": StressLevel.MODERATE_STRESS,
    "angry": StressLevel.MODERATE_STRESS,
    "disgust": StressLevel.SEVERE_STRESS,
}

# The two tables disagree on Disgust (Moderate for faces, Severe for speech)
# and on Angry (Severe for faces, Moderate for speech) -- keep them visible
# here so the disagreement is never silently lost.
KNOWN_CROSS_MODALITY_DISAGREEMENTS = {
    "disgust/Disgust": (FACIAL_EMOTION_TO_STRESS["Disgust"], SPEECH_EMOTION_TO_STRESS["disgust"]),
    "angry/Angry": (FACIAL_EMOTION_TO_STRESS["Angry"], SPEECH_EMOTION_TO_STRESS["angry"]),
}


def facial_logits_to_stress_distribution(emotion_probs: dict[str, float]) -> list[float]:
    """Project a 7-way facial emotion probability distribution onto the 4-tier
    severity axis by summing probability mass per stress level.

    `emotion_probs` keys must be FER emotion names (see schemas.FACIAL_EMOTIONS).
    Returns a length-4 list of stress-level probabilities, index-aligned to
    StressLevel (0=Healthy .. 3=Severe).
    """
    dist = [0.0, 0.0, 0.0, 0.0]
    for emotion, prob in emotion_probs.items():
        level = FACIAL_EMOTION_TO_STRESS[emotion]
        dist[int(level)] += prob
    return dist


def speech_logits_to_stress_distribution(emotion_probs: dict[str, float]) -> list[float]:
    """Same projection as above, for the 8-way RAVDESS speech emotion vocabulary."""
    dist = [0.0, 0.0, 0.0, 0.0]
    for emotion, prob in emotion_probs.items():
        level = SPEECH_EMOTION_TO_STRESS[emotion]
        dist[int(level)] += prob
    return dist
