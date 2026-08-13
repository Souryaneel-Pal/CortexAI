import base64
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app


CLINICIAN_CREDENTIALS = {"email": "julian.vance@cortex.ai", "password": "password"}
ADMIN_CREDENTIALS = {"email": "admin", "password": "admin"}


@pytest.fixture(scope="module")
def anon_client():
    """Unauthenticated client, for asserting the routes are actually protected."""
    with TestClient(app) as c:
        yield c


def _login(client: TestClient, credentials: dict) -> str:
    response = client.post("/api/auth/login", json=credentials)
    assert response.status_code == 200, response.text
    return response.json()["token"]


@pytest.fixture(scope="module")
def client():
    """Client authenticated as the demo Clinician.

    It logs in for real rather than bypassing `get_current_user`: the app used
    to short-circuit auth whenever `pytest` was importable, which meant every
    protected route was reachable unauthenticated in tests and the guards were
    never actually exercised. Going through `/api/auth/login` means the token
    issuing, signing, and verification path is covered by every API test.
    """
    with TestClient(app) as c:
        token = _login(c, CLINICIAN_CREDENTIALS)
        c.headers.update({"Authorization": f"Bearer {token}"})
        yield c


@pytest.fixture(scope="module")
def admin_client():
    with TestClient(app) as c:
        token = _login(c, ADMIN_CREDENTIALS)
        c.headers.update({"Authorization": f"Bearer {token}"})
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
    # Whether checkpoints exist depends on whether training has been run, so
    # this asserts the flag is *correct*, not that it has one fixed value:
    # it must be False exactly when every served module loaded real weights.
    # (An earlier version hardcoded `is True`, which encoded "this sandbox has
    # no checkpoints" and started failing the moment the models were trained.)
    from src.api.main import pipeline

    assert body["is_demo_untrained_model"] == (not all(pipeline.loaded_modules.values()))


def test_assess_tabular_only(client):
    response = client.post("/assess", json={"tabular_features": _sample_tabular_features()})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in ("Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress")
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["scores"].keys()) == {"Depression_Score", "Anxiety_Score", "Stress_Score"}
    assert abs(sum(body["modality_weights"].values()) - 1.0) < 1e-3
    assert isinstance(body["is_demo_untrained_model"], bool)
    assert "decision-support" in body["disclaimer"].lower()

    # Scores must land inside their documented ranges regardless of training
    # state -- RegressionHead scales a sigmoid by each target's max, so an
    # out-of-range score means that clamping regressed.
    assert 0.0 <= body["scores"]["Depression_Score"] <= 34.0
    assert 0.0 <= body["scores"]["Anxiety_Score"] <= 24.0
    assert 0.0 <= body["scores"]["Stress_Score"] <= 39.0

    # Tabular-only request: no face or clip was sent, so the emotion
    # distributions must be empty rather than a softmax over a zero tensor.
    assert body["face_emotion_probs"] == {}
    assert body["speech_emotion_probs"] == {}


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
    assert len(explain_body["top_shap_features"]) <= 8

    # Signed SHAP covers all 18 features and carries direction, which the
    # mean|SHAP| ranking discards -- the UI's diverging bar chart needs it.
    assert len(explain_body["signed_shap"]) == 18
    assert any(f["shap"] < 0 for f in explain_body["signed_shap"]) or any(
        f["shap"] > 0 for f in explain_body["signed_shap"]
    )

    # Grad-CAM: a face was supplied, so a displayable overlay must come back.
    gradcam = explain_body["gradcam"]
    assert gradcam is not None
    assert len(gradcam["heatmap"]) == 48 and len(gradcam["heatmap"][0]) == 48
    assert all(0.0 <= v <= 1.0 for row in gradcam["heatmap"] for v in row)
    assert base64.b64decode(gradcam["overlay_png_base64"])[:8] == b"\x89PNG\r\n\x1a\n"

    # Integrated Gradients over the waveform, pooled to frames.
    audio_ig = explain_body["audio_integrated_gradients"]
    assert audio_ig is not None
    assert audio_ig["frame_importance"]
    assert all(0.0 <= v <= 1.0 for v in audio_ig["frame_importance"])

    # Both modalities present, so the Masked-Distress Index is computable.
    mdi = explain_body["masked_distress_index"]
    assert 0.0 <= mdi["mdi"] <= 1.0
    assert "unavailable_reason" not in mdi

    weights = explain_body["modality_weights"]
    assert abs(weights["face"] + weights["speech"] + weights["tabular"] - 1.0) < 1e-4

    report_response = client.get(f"/report/{session_id}")
    assert report_response.status_code == 200
    report_body = report_response.json()
    # `cached` depends on whether a local Ollama is up on this machine, so
    # assert the invariant instead of one environment's answer: a templated
    # report always states why it degraded, a live one never claims to have.
    assert isinstance(report_body["cached"], bool)
    assert (report_body["fallback_reason"] is not None) == report_body["cached"]
    # Either way the report is cited and carries the responsible-AI framing.
    assert report_body["citations"]
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


def test_predict_alias_matches_assess(client):
    """`/predict` and `/assess` are the same handler; clients reach for either."""
    payload = {"tabular_features": _sample_tabular_features()}

    assess = client.post("/assess", json=payload)
    predict = client.post("/predict", json=payload)

    assert assess.status_code == predict.status_code == 200
    assert set(assess.json().keys()) == set(predict.json().keys())
    # Same schema and same class; session_id differs per call by design.
    assert predict.json()["predicted_class"] in ("Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress")


def test_health_reports_local_reasoning_stack(client):
    """The UI warns before an assessment that the narrative may be templated,
    so /health has to describe the local Ollama stack."""
    body = client.get("/health").json()
    reasoning = body["reasoning"]

    assert set(reasoning) >= {
        "ollama_reachable",
        "base_url",
        "llm_model",
        "llm_available",
        "embedding_model",
        "embedding_available",
        "retrieval_backend",
    }
    assert isinstance(reasoning["ollama_reachable"], bool)
    assert reasoning["llm_model"] == "llama3.1"
    assert reasoning["embedding_model"] == "nomic-embed-text"
    # Retrieval always resolves to *some* working backend.
    assert reasoning["retrieval_backend"] in ("ollama", "sentence_transformers", "tfidf")
    # When a model is missing the detail must be actionable, not just a flag.
    if not reasoning["llm_available"] or not reasoning["embedding_available"]:
        assert reasoning["detail"]


def test_report_declares_its_generator(client):
    """A templated fallback must never be mistaken for a model-written
    narrative: the response says which generator ran, and why if it degraded."""
    session_id = client.post("/assess", json={"tabular_features": _sample_tabular_features()}).json()["session_id"]
    body = client.get(f"/report/{session_id}").json()

    assert body["generator"] in ("template",) or body["generator"].startswith(("ollama:", "anthropic:"))
    # cached=True means templated, which must always carry a stated reason.
    assert body["cached"] is (body["generator"] == "template")
    if body["cached"]:
        assert body["fallback_reason"]
    assert body["citations"]


def test_base64_data_uri_decoding():
    from src.api.inference import _decode_face_image, _decode_speech_audio
    import base64
    import io
    from PIL import Image
    import numpy as np
    import soundfile as sf

    # Create a simple 48x48 dummy image in memory
    img = Image.new('L', (48, 48), color=128)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    img_data_uri = f"data:image/png;base64,{img_b64}"

    # Verify _decode_face_image handles Data URI prefix
    face_tensor = _decode_face_image(img_data_uri)
    assert face_tensor.shape == (1, 1, 48, 48)

    # For audio, create a minimal valid WAV file (100ms at 16000Hz)
    audio_data = np.zeros(1600, dtype=np.float32)
    audio_byte_arr = io.BytesIO()
    sf.write(audio_byte_arr, audio_data, 16000, format='WAV', subtype='PCM_16')
    audio_bytes = audio_byte_arr.getvalue()
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    audio_data_uri = f"data:audio/wav;base64,{audio_b64}"

    # Verify _decode_speech_audio handles Data URI prefix
    speech_tensor = _decode_speech_audio(audio_data_uri)
    assert speech_tensor.shape == (1, 64000)



# ---------------------------------------------------------------------------
# Authentication and the Admin-only settings guard.
#
# None of this was reachable before: `get_current_user` returned a hardcoded
# Admin whenever pytest was in sys.modules, so the guards below were dead code
# under test. They are the security surface of the app, so they get pinned.
# ---------------------------------------------------------------------------
PROTECTED_ROUTES = [
    ("get", "/api/settings"),
    ("get", "/api/dashboard"),
    ("get", "/api/analytics"),
]


@pytest.mark.parametrize(("method", "path"), PROTECTED_ROUTES)
def test_protected_routes_reject_anonymous_callers(anon_client, method, path):
    response = getattr(anon_client, method)(path)
    assert response.status_code == 401, f"{path} served an unauthenticated caller"


def test_assessment_endpoints_reject_anonymous_callers(anon_client):
    response = anon_client.post("/predict", json={"tabular_features": _sample_tabular_features()})
    assert response.status_code == 401


def test_login_rejects_bad_credentials(anon_client):
    response = anon_client.post(
        "/api/auth/login", json={"email": "admin", "password": "not-the-password"}
    )
    assert response.status_code == 401
    assert "token" not in response.json()


def test_login_returns_role_for_each_demo_account(anon_client):
    clinician = anon_client.post("/api/auth/login", json=CLINICIAN_CREDENTIALS).json()
    assert clinician["user"]["role"] == "Clinician"
    assert clinician["token"]

    admin = anon_client.post("/api/auth/login", json=ADMIN_CREDENTIALS).json()
    assert admin["user"]["role"] == "Admin"


def test_tampered_token_is_rejected(anon_client):
    token = _login(anon_client, ADMIN_CREDENTIALS)
    header, payload, signature = token.split(".")

    # Re-sign nothing: flip the payload but keep the original signature. The
    # HMAC check must catch it -- otherwise anyone could self-promote to Admin.
    import base64 as _b64
    import json as _json

    decoded = _json.loads(_b64.urlsafe_b64decode(payload + "=" * (4 - len(payload) % 4)))
    decoded["role"] = "Admin"
    decoded["sub"] = "attacker@example.com"
    forged_payload = _b64.urlsafe_b64encode(_json.dumps(decoded).encode()).rstrip(b"=").decode()
    forged = f"{header}.{forged_payload}.{signature}"

    response = anon_client.get("/api/settings", headers={"Authorization": f"Bearer {forged}"})
    assert response.status_code == 401


def test_settings_readable_by_clinician(client):
    body = client.get("/api/settings").json()
    assert set(body) == {
        "uncertainty_threshold",
        "mdi_threshold",
        "ignore_face",
        "ignore_speech",
        "ignore_tabular",
    }


def test_settings_write_is_admin_only(client, admin_client):
    payload = {
        "uncertainty_threshold": 0.75,
        "mdi_threshold": 0.55,
        "ignore_face": False,
        "ignore_speech": False,
        "ignore_tabular": False,
    }

    # A Clinician may read settings but must not change model behaviour.
    assert client.put("/api/settings", json=payload).status_code == 403

    original = admin_client.get("/api/settings").json()
    try:
        assert admin_client.put("/api/settings", json=payload).status_code == 200
        assert admin_client.get("/api/settings").json()["uncertainty_threshold"] == 0.75
    finally:
        admin_client.put("/api/settings", json=original)


def test_persisted_report_is_not_relabelled_as_a_template(client):
    """Re-reading a stored report must not turn a model-written narrative into
    a "Templated summary".

    `cached` means "this is a templated fallback, not model-written" -- the UI
    keys its "Narrative not model-generated" warning off it. The DB-cache path
    used to hardcode `cached=True` for every stored row, so the second view of
    an `ollama:llama3.1` narrative (the normal path once persistence exists)
    claimed it was a template. `from_store` carries the "served from SQLite"
    fact separately.
    """
    session_id = client.post("/assess", json={"tabular_features": _sample_tabular_features()}).json()["session_id"]

    first = client.get(f"/report/{session_id}").json()
    second = client.get(f"/report/{session_id}").json()

    # Same narrative both times, and the generator attribution is stable.
    assert first["narrative"] == second["narrative"]
    assert first["generator"] == second["generator"]

    # The stored read is flagged as such, and `cached` still tracks only
    # "is this a template", never "did this come from the database".
    assert second["from_store"] is True
    assert second["cached"] == (second["generator"] == "template")
    if second["generator"].startswith("ollama:"):
        assert second["cached"] is False, "a stored llama3.1 narrative is not a template"
        assert second["fallback_reason"] is None
