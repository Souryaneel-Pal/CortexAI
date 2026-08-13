import base64
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _sample_tabular_features() -> dict:
    return {
        "Sleep_Quality": 3.0,
        "Social_Engagement": 3.0,
        "Daily_App_Usage_Min": 120.0,
        "Typing_Speed_WPM": 40.0,
        "Session_Frequency": 8.0,
        "Idle_Time_Min": 60.0,
        "Facial_Emotion_Variance": 0.5,
        "Eye_Blink_Rate": 15.0,
        "Smile_Intensity": 0.3,
        "Head_Motion_Index": 0.2,
        "MFCC_Mean": 0.1,
        "MFCC_Variance": 0.05,
        "Pitch_Mean": 150.0,
        "Speech_Rate": 3.5,
        "Heart_Rate_BPM": 72.0,
        "HRV_Index": 50.0,
        "Skin_Temperature": 33.0,
        "GSR_Level": 2.0,
    }


def _sample_face_b64() -> str:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(48, 48), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _sample_speech_b64() -> str:
    rng = np.random.default_rng(0)
    waveform = rng.uniform(-0.1, 0.1, size=16000).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, 16000, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["is_demo_untrained_model"] is True  # no checkpoints in this sandbox


def test_assess_tabular_only(client):
    response = client.post("/assess", json={"tabular_features": _sample_tabular_features()})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in ("Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress")
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["scores"].keys()) == {"Depression_Score", "Anxiety_Score", "Stress_Score"}
    assert abs(sum(body["modality_weights"].values()) - 1.0) < 1e-3
    assert body["is_demo_untrained_model"] is True
    assert "decision-support" in body["disclaimer"].lower()


def test_assess_with_all_three_modalities(client):
    response = client.post(
        "/assess",
        json={
            "tabular_features": _sample_tabular_features(),
            "face_image_base64": _sample_face_b64(),
            "speech_audio_base64": _sample_speech_b64(),
        },
    )
    assert response.status_code == 200


def test_full_session_flow_assess_explain_report_followup(client):
    assess_response = client.post(
        "/assess",
        json={
            "tabular_features": _sample_tabular_features(),
            "face_image_base64": _sample_face_b64(),
            "speech_audio_base64": _sample_speech_b64(),
        },
    )
    session_id = assess_response.json()["session_id"]

    explain_response = client.get(f"/explain/{session_id}")
    assert explain_response.status_code == 200
    explain_body = explain_response.json()
    assert len(explain_body["top_shap_features"]) <= 5

    report_response = client.get(f"/report/{session_id}")
    assert report_response.status_code == 200
    report_body = report_response.json()
    assert report_body["cached"] is True  # no ANTHROPIC_API_KEY in this sandbox
    assert "decision-support" in report_body["disclaimer"].lower()

    followup_response = client.post(
        "/follow-up", json={"session_id": session_id, "question": "why this result?"}
    )
    assert followup_response.status_code == 200
    assert followup_response.json()["answer"]


def test_unknown_session_returns_404(client):
    response = client.get("/explain/does-not-exist")
    assert response.status_code == 404


def test_counterfactual_endpoint(client):
    assess_response = client.post("/assess", json={"tabular_features": _sample_tabular_features()})
    session_id = assess_response.json()["session_id"]

    cf_response = client.get(f"/counterfactual/{session_id}")
    assert cf_response.status_code == 200
    # counterfactual may legitimately be None (already at target, or no
    # actionable feature flips it) -- just check the response is well-formed.
    body = cf_response.json()
    assert "counterfactual" in body and "narrative" in body
