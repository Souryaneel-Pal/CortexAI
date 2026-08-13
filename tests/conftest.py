"""Synthetic, schema-correct fixtures for the three modalities -- generated on
the fly so no binary media is committed to the repo. Shapes, column names,
and the RAVDESS filename convention match docs/1.docx exactly; only the
content is fake, never presented as real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schemas import (
    FACIAL_EMOTIONS,
    FACIAL_IMAGE_SIZE,
    SPEECH_EMOTIONS,
    TABULAR_FEATURE_COLUMNS,
    TABULAR_TARGET_CLASS_COLUMN,
    TABULAR_TARGET_SCORE_COLUMNS,
)


@pytest.fixture
def synthetic_facial_dir(tmp_path):
    root = tmp_path / "facial"
    root.mkdir()
    from PIL import Image

    rng = np.random.default_rng(0)
    for emotion in FACIAL_EMOTIONS.values():
        class_dir = root / emotion
        class_dir.mkdir()
        for i in range(3):
            arr = rng.integers(0, 255, size=(FACIAL_IMAGE_SIZE, FACIAL_IMAGE_SIZE), dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(class_dir / f"{emotion}_{i}.png")
    return root


@pytest.fixture
def synthetic_speech_dir(tmp_path):
    root = tmp_path / "speech"
    root.mkdir()
    import soundfile as sf

    rng = np.random.default_rng(0)
    sr = 16000
    actor = 1
    for emotion_code in SPEECH_EMOTIONS:
        intensities = ["01"] if emotion_code == "01" else ["01", "02"]
        for intensity in intensities:
            waveform = rng.uniform(-0.1, 0.1, size=sr // 4).astype(np.float32)
            fname = f"03-01-{emotion_code}-{intensity}-01-01-{actor:02d}.wav"
            sf.write(str(root / fname), waveform, sr)
        actor += 1
    return root


@pytest.fixture
def synthetic_tabular_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    classes = ["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"]
    data = {col: rng.normal(size=n) for col in TABULAR_FEATURE_COLUMNS}
    data[TABULAR_TARGET_CLASS_COLUMN] = rng.choice(classes, size=n)
    data[TABULAR_TARGET_SCORE_COLUMNS[0]] = rng.uniform(0, 34, size=n)
    data[TABULAR_TARGET_SCORE_COLUMNS[1]] = rng.uniform(0, 24, size=n)
    data[TABULAR_TARGET_SCORE_COLUMNS[2]] = rng.uniform(0, 39, size=n)
    df = pd.DataFrame(data)
    csv_path = tmp_path / "numerical.csv"
    df.to_csv(csv_path, index=False)
    return csv_path
