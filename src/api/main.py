"""CortexAI FastAPI backend (docs/PROJECT_PLAN.md P5).

Endpoints: POST /assess (alias: POST /predict), GET /explain/{session_id},
GET /counterfactual/{session_id}, GET /report/{session_id}, POST /follow-up,
GET /health.

`/predict` and `/assess` are the same handler. `/assess` is the original
name and what the tests and frontend use; `/predict` is registered as an
alias because it is the more conventional name for this operation and
clients reasonably reach for it first.

`/health` also reports the state of the local Ollama reasoning stack
(`reasoning`), so a client can tell before running an assessment whether the
narrative will be model-written or a templated fallback.

Session state (the AgentState produced by src/reasoning/agent_graph.py) is
kept in a simple in-memory dict keyed by session_id so /explain, /report,
and /follow-up can reuse a prior /assess call without recomputing the
prediction. This is an intentionally minimal MVP store (docs/MINDSCOPE_Blueprint.pdf's
full tech-stack table names Redis/Postgres for this in production) -- swap
`SESSION_STORE` for a real backing store before any multi-instance deploy.
"""
from __future__ import annotations

import logging
import uuid
import datetime
import json
import os
import random
import hmac
import hashlib
import base64
import time
import torch

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.inference import CortexAIPipeline, UnsupportedAudioFormatError
from src.api.schemas import (
    AssessmentRequest,
    CounterfactualResponse,
    ExplanationResponse,
    FollowUpRequest,
    FollowUpResponse,
    ModalityWeights,
    PredictionResponse,
    ReasoningStatus,
    ReportResponse,
    ScoreBreakdown,
    UncertaintyInfo,
)
from src.reasoning.ollama_config import embedding_model, llm_model, probe_ollama
from src.data.schemas import StressLevel
from src.explain.counterfactual import format_counterfactual_narrative, generate_counterfactual_grid_search
from src.reasoning.agent_graph import answer_follow_up, build_agent_graph

# --- Security / JWT Helpers ---
# Read from the environment so a deployment can set a real secret. The default
# is a development-only value: with it, anyone who has the source can mint a
# valid token, so `CORTEXAI_JWT_SECRET` must be set before any real use.
SECRET_KEY = os.environ.get("CORTEXAI_JWT_SECRET", "dev-only-insecure-cortexai-signing-key")

def base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')

def base64url_decode(data: str) -> bytes:
    padding = '=' * (4 - (len(data) % 4))
    return base64.urlsafe_b64decode(data + padding)

def create_jwt(payload: dict) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = base64url_encode(json.dumps(header).encode('utf-8'))
    payload_b64 = base64url_encode(json.dumps(payload).encode('utf-8'))
    signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
    signature = hmac.new(SECRET_KEY.encode('utf-8'), signing_input, hashlib.sha256).digest()
    signature_b64 = base64url_encode(signature)
    return f"{header_b64}.{payload_b64}.{signature_b64}"

def verify_jwt(token: str) -> dict | None:
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None
        header_b64, payload_b64, signature_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
        expected_sig = hmac.new(SECRET_KEY.encode('utf-8'), signing_input, hashlib.sha256).digest()
        expected_sig_b64 = base64url_encode(expected_sig)
        if not hmac.compare_digest(signature_b64, expected_sig_b64):
            return None
        payload = json.loads(base64url_decode(payload_b64).decode('utf-8'))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None

def get_current_user(authorization: str = Header(None)) -> dict:
    """Resolve the caller from a Bearer token, or reject with 401.

    This deliberately has **no test-mode bypass**. An earlier version returned
    a hardcoded Admin whenever `pytest` was in `sys.modules`, which meant the
    entire authentication and the Admin-only guard on `PUT /api/settings` were
    unreachable in tests -- the one behaviour most worth pinning was the one
    thing no test could exercise, and a stray `import pytest` anywhere in a
    production process would have disabled auth outright.

    Tests authenticate for real against the demo accounts, or override this
    dependency explicitly via `app.dependency_overrides` (see tests/test_api.py),
    which keeps the bypass in the test suite where it belongs.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    payload = verify_jwt(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload

# --- Dynamic Settings and Auth request schemas ---
class LoginRequest(BaseModel):
    email: str
    password: str

class SettingsUpdate(BaseModel):
    uncertainty_threshold: float
    mdi_threshold: float
    ignore_face: bool
    ignore_speech: bool
    ignore_tabular: bool

app = FastAPI(
    title="CortexAI API",
    description="Explainable multimodal mental-health screening -- decision support, not diagnosis.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

pipeline: CortexAIPipeline | None = None
agent_app = None
SESSION_STORE: dict[str, dict] = {}


@app.exception_handler(UnsupportedAudioFormatError)
def _unsupported_audio_handler(request, exc: UnsupportedAudioFormatError):
    """Answer 400 with one actionable sentence.

    Without this the decoder's failure propagated as an unhandled exception, so
    the client got a 500 and the server log filled with the libtorchcodec
    loader traceback -- roughly 400 lines listing a failure for every FFmpeg
    major version it probes, which buries the single line that matters. An
    undecodable upload is a bad request, not a server fault.
    """
    from fastapi.responses import JSONResponse

    logger.warning("Rejected an undecodable audio upload: %s", exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.on_event("startup")
def _startup() -> None:
    global pipeline, agent_app
    from src.api.database import init_db
    init_db()
    pipeline = CortexAIPipeline()
    agent_app = build_agent_graph(pipeline.context)


def _reasoning_status() -> ReasoningStatus:
    """Live view of the local Ollama stack.

    Reported on /health so the UI can warn *before* an assessment that the
    narrative will be templated, rather than surprising the clinician with a
    fallback notice after they have already run one.
    """
    status = probe_ollama()
    llm, embed = llm_model(), embedding_model()
    detail = status.unavailable_reason(llm) or status.unavailable_reason(embed)
    return ReasoningStatus(
        ollama_reachable=status.reachable,
        base_url=status.base_url,
        llm_model=llm,
        llm_available=status.has_model(llm),
        embedding_model=embed,
        embedding_available=status.has_model(embed),
        retrieval_backend=pipeline.retriever.active_backend if pipeline else None,
        vector_index=pipeline.retriever.index_impl if pipeline else None,
        detail=detail,
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "is_demo_untrained_model": pipeline.is_demo_untrained_model if pipeline else None,
        "reasoning": _reasoning_status().model_dump(),
    }

# --- Auth & Settings Endpoints ---

@app.post("/api/auth/login")
def login(request: LoginRequest):
    email = request.email.strip().lower()
    password = request.password
    
    # Demo accounts. These are hardcoded on purpose for the hackathon build --
    # there is no user store, and the credentials are documented in README.md.
    # Replace with a real identity provider before any deployment: see the
    # "Authentication" section of the README for what this is and is not.
    if email == "admin" and password == "admin":
        role = "Admin"
        name = "System Administrator"
    elif email == "julian.vance@cortex.ai" and password == "password":
        role = "Clinician"
        name = "Dr. Julian Vance"
    else:
        raise HTTPException(status_code=401, detail="Invalid email or password")
        
    payload = {
        "sub": email,
        "name": name,
        "role": role,
        "exp": int(time.time()) + 86400  # 24 hours
    }
    token = create_jwt(payload)
    return {
        "token": token,
        "user": {
            "email": email,
            "name": name,
            "role": role
        }
    }


@app.get("/api/settings")
def get_settings(current_user: dict = Depends(get_current_user)):
    from src.api.database import get_db_setting
    return {
        "uncertainty_threshold": get_db_setting("uncertainty_threshold", 0.60),
        "mdi_threshold": get_db_setting("mdi_threshold", 0.50),
        "ignore_face": get_db_setting("ignore_face", False),
        "ignore_speech": get_db_setting("ignore_speech", False),
        "ignore_tabular": get_db_setting("ignore_tabular", False),
    }


@app.put("/api/settings")
def update_settings(settings: SettingsUpdate, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "Admin":
        raise HTTPException(status_code=403, detail="Restricted to Admin users")
    from src.api.database import set_db_setting
    set_db_setting("uncertainty_threshold", settings.uncertainty_threshold)
    set_db_setting("mdi_threshold", settings.mdi_threshold)
    set_db_setting("ignore_face", settings.ignore_face)
    set_db_setting("ignore_speech", settings.ignore_speech)
    set_db_setting("ignore_tabular", settings.ignore_tabular)
    return {"status": "ok"}


@app.get("/api/dashboard")
def get_dashboard(current_user: dict = Depends(get_current_user)):
    from src.api.dashboard_metrics import derive_dashboard_metrics
    return derive_dashboard_metrics()


@app.get("/api/analytics")
def get_analytics(current_user: dict = Depends(get_current_user)):
    """Cohort analytics, derived entirely from stored assessments.

    See `src/api/dashboard_metrics.derive_analytics_metrics` for the list of
    hardcoded values this replaced -- including a fabricated "AI Efficacy
    Score: 94%" that contradicted the project's own measured macro-F1 of 0.228.
    """
    from src.api.dashboard_metrics import derive_analytics_metrics

    return derive_analytics_metrics()


# --- Assessment & Explain Endpoints ---

@app.post("/assess", response_model=PredictionResponse)
@app.post("/predict", response_model=PredictionResponse)
def assess(request: AssessmentRequest, current_user: dict = Depends(get_current_user)) -> PredictionResponse:
    raw_input = {
        "tabular_features": request.tabular_features.to_ordered_list(),
        "face_image_base64": request.face_image_base64,
        "speech_audio_base64": request.speech_audio_base64,
    }

    session_id = str(uuid.uuid4())
    # Stop graph execution after explanations (so report RAG generation is skipped)
    final_state = agent_app.invoke({"raw_input": raw_input, "log": [], "generate_report": False})
    SESSION_STORE[session_id] = final_state

    prediction = final_state["prediction"]
    uncertainty = final_state["uncertainty"]

    mask = final_state["preprocessed"]["modality_mask"]
    
    # Save the assessment run to the database
    from src.api.database import SessionLocal, DBAssessment
    db = SessionLocal()
    try:
        patient_id = (request.patient_id.strip() if request.patient_id else None) or f"PT-{random.randint(1000, 9999)}"
        demographic = (request.demographic.strip() if request.demographic else None) or random.choice(["Adults", "Seniors", "Adolescents"])
        
        db_assess = DBAssessment(
            session_id=session_id,
            # Naive UTC to match the DateTime column and the dashboard parser;
            # utcnow() itself is deprecated for removal in 3.12+.
            completed_at=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
            patient_id=patient_id,
            demographic=demographic,
            tabular_features=json.dumps(request.tabular_features.model_dump()),
            face_image_base64=request.face_image_base64,
            speech_audio_base64=request.speech_audio_base64,
            predicted_class=pipeline.predicted_class_name(prediction["predicted_class"]),
            confidence=prediction["confidence"],
            class_probs=json.dumps(prediction["class_probs"]),
            depression_score=prediction["scores"]["Depression_Score"],
            anxiety_score=prediction["scores"]["Anxiety_Score"],
            stress_score=prediction["scores"]["Stress_Score"],
            modality_weights=json.dumps(prediction["modality_weights"]),
            face_emotion_probs=json.dumps(prediction["face_emotion_probs"] if bool(mask["face"].item()) else {}),
            speech_emotion_probs=json.dumps(prediction["speech_emotion_probs"] if bool(mask["speech"].item()) else {}),
            deferred_to_human=uncertainty["defer"],
            shap_ranked_features=json.dumps(final_state["explanations"]["shap_ranked_features"]),
            signed_shap=json.dumps(final_state["explanations"]["signed_shap"]),
            masked_distress_index=json.dumps(final_state["explanations"]["masked_distress_index"]),
            gradcam=json.dumps(final_state["explanations"].get("gradcam")),
            audio_integrated_gradients=json.dumps(final_state["explanations"].get("audio_integrated_gradients")),
            report_narrative=None,
            report_citations=None,
            report_generator=None,
            report_fallback_reason=None
        )
        db.add(db_assess)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error saving assessment: {e}")
    finally:
        db.close()

    return PredictionResponse(
        session_id=session_id,
        predicted_class=pipeline.predicted_class_name(prediction["predicted_class"]),
        confidence=prediction["confidence"],
        class_probs=prediction["class_probs"],
        scores=ScoreBreakdown(**prediction["scores"]),
        modality_weights=ModalityWeights(**prediction["modality_weights"]),
        face_emotion_probs=prediction["face_emotion_probs"] if bool(mask["face"].item()) else {},
        speech_emotion_probs=prediction["speech_emotion_probs"] if bool(mask["speech"].item()) else {},
        uncertainty=UncertaintyInfo(**uncertainty),
        deferred_to_human=uncertainty["defer"],
        is_demo_untrained_model=pipeline.is_demo_untrained_model,
    )


def _get_session_or_reconstruct(session_id: str) -> dict:
    state = SESSION_STORE.get(session_id)
    if state is not None:
        return state
        
    from src.api.database import SessionLocal, DBAssessment
    db = SessionLocal()
    db_assess = None
    try:
        db_assess = db.query(DBAssessment).filter(DBAssessment.session_id == session_id).first()
    finally:
        db.close()
        
    if db_assess is None:
        raise HTTPException(status_code=404, detail="Unknown session_id -- call /assess first.")
        
    # Reconstruct the state dictionary from database record
    try:
        classes = ["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"]
        class_idx = classes.index(db_assess.predicted_class) if db_assess.predicted_class in classes else 0
    except ValueError:
        class_idx = 0
        
    state = {
        "prediction": {
            "predicted_class": class_idx,
            "confidence": db_assess.confidence,
            "class_probs": json.loads(db_assess.class_probs) if db_assess.class_probs else [0.25]*4,
            "scores": {
                "Depression_Score": db_assess.depression_score,
                "Anxiety_Score": db_assess.anxiety_score,
                "Stress_Score": db_assess.stress_score
            },
            "modality_weights": json.loads(db_assess.modality_weights) if db_assess.modality_weights else {},
            "face_emotion_probs": json.loads(db_assess.face_emotion_probs) if db_assess.face_emotion_probs else {},
            "speech_emotion_probs": json.loads(db_assess.speech_emotion_probs) if db_assess.speech_emotion_probs else {}
        },
        "uncertainty": {
            "defer": db_assess.deferred_to_human
        },
        "explanations": {
            "shap_ranked_features": json.loads(db_assess.shap_ranked_features) if db_assess.shap_ranked_features else [],
            "signed_shap": json.loads(db_assess.signed_shap) if db_assess.signed_shap else [],
            "masked_distress_index": json.loads(db_assess.masked_distress_index) if db_assess.masked_distress_index else None,
            "gradcam": json.loads(db_assess.gradcam) if db_assess.gradcam else None,
            "audio_integrated_gradients": json.loads(db_assess.audio_integrated_gradients) if db_assess.audio_integrated_gradients else None
        },
        "preprocessed": {
            "tabular_raw": torch.zeros(1, 18),
            "modality_mask": {
                "face": torch.tensor([db_assess.face_image_base64 is not None]),
                "speech": torch.tensor([db_assess.speech_audio_base64 is not None])
            }
        }
    }
    
    SESSION_STORE[session_id] = state
    return state


@app.get("/explain/{session_id}", response_model=ExplanationResponse)
def explain(session_id: str, current_user: dict = Depends(get_current_user)) -> ExplanationResponse:
    state = _get_session_or_reconstruct(session_id)
    explanations = state["explanations"]
    return ExplanationResponse(
        session_id=session_id,
        top_shap_features=explanations["shap_ranked_features"][:8],
        signed_shap=explanations.get("signed_shap", []),
        modality_weights=ModalityWeights(**state["prediction"]["modality_weights"]),
        masked_distress_index=explanations.get("masked_distress_index"),
        gradcam=explanations.get("gradcam"),
        audio_integrated_gradients=explanations.get("audio_integrated_gradients"),
        is_demo_untrained_model=pipeline.is_demo_untrained_model,
    )


@app.get("/counterfactual/{session_id}", response_model=CounterfactualResponse)
def counterfactual(session_id: str, target_class: int | None = None, current_user: dict = Depends(get_current_user)) -> CounterfactualResponse:
    state = _get_session_or_reconstruct(session_id)
    prediction = state["prediction"]
    current_class = prediction["predicted_class"]

    if target_class is None:
        target_class = max(int(current_class) - 1, int(StressLevel.HEALTHY))
    if target_class == current_class:
        return CounterfactualResponse(session_id=session_id, counterfactual=None, narrative=None)

    from src.data.schemas import TABULAR_FEATURE_COLUMNS

    tabular = state["preprocessed"]["tabular_raw"][0].tolist()
    query = dict(zip(TABULAR_FEATURE_COLUMNS, tabular))
    actionable_features = ["Sleep_Quality", "Social_Engagement", "Daily_App_Usage_Min", "Session_Frequency"]
    ranges = {f: (0.0, 5.0) if f == "Sleep_Quality" or f == "Social_Engagement" else (0.0, 300.0) for f in actionable_features}

    result = generate_counterfactual_grid_search(
        pipeline.raw_unit_tabular_model,
        query_instance=query,
        feature_ranges=ranges,
        desired_class=target_class,
        features_to_vary=actionable_features,
    )
    narrative = format_counterfactual_narrative(result) if result else None
    return CounterfactualResponse(session_id=session_id, counterfactual=result, narrative=narrative)


@app.get("/report/{session_id}", response_model=ReportResponse)
def report(session_id: str, current_user: dict = Depends(get_current_user)) -> ReportResponse:
    from src.api.database import SessionLocal, DBAssessment
    db = SessionLocal()
    db_assess = None
    try:
        db_assess = db.query(DBAssessment).filter(DBAssessment.session_id == session_id).first()
    finally:
        db.close()
        
    if db_assess is None:
        raise HTTPException(status_code=404, detail="Unknown session_id -- call /assess first.")
        
    # Already generated for this session -- serve the stored narrative rather
    # than paying for a second LLM run over identical inputs.
    if db_assess.report_narrative is not None:
        stored_generator = db_assess.report_generator or "template"
        return ReportResponse(
            session_id=session_id,
            narrative=db_assess.report_narrative,
            citations=json.loads(db_assess.report_citations) if db_assess.report_citations else [],
            # `cached` means "this is a templated fallback, not model-written" --
            # it is what the UI keys the "Narrative not model-generated" warning
            # off. It must NOT be set merely because the row came back from
            # SQLite: hardcoding cached=True here relabelled every persisted
            # llama3.1 narrative as a "Templated summary" the second time it was
            # viewed, which (after persistence landed) is the normal path. The
            # stored generator is the source of truth for that distinction;
            # `from_store` carries the "served from the database" fact instead.
            cached=stored_generator == "template",
            generator=stored_generator,
            fallback_reason=db_assess.report_fallback_reason,
            from_store=True,
        )

    # Reconstruct state and execute RAG report generation on demand
    state = _get_session_or_reconstruct(session_id)
    
    # Run retrieve logic
    prediction = state["prediction"]
    explanations = state.get("explanations", {})
    query_parts = [f"stress status {prediction.get('predicted_class')}"]
    top_feature = explanations.get("top_shap_feature")
    if not top_feature and explanations.get("shap_ranked_features"):
        top_feature = explanations["shap_ranked_features"][0]["feature"]
    if top_feature:
        query_parts.append(str(top_feature))
    query = " ".join(query_parts)

    docs = pipeline.retriever.retrieve(query, k=pipeline.context.top_k_docs)
    
    # Crisis resources check
    from src.api.database import get_db_setting
    mdi_thresh = get_db_setting("mdi_threshold", 0.50)
    mdi = explanations.get("masked_distress_index", {}).get("mdi", 0.0)
    
    severe = prediction.get("predicted_class") == 3
    masked_distress_flagged = mdi >= mdi_thresh
    if severe or masked_distress_flagged:
        existing_ids = {d["id"] for d in docs}
        for doc in pipeline.retriever.get_by_category("crisis_resource"):
            if doc["id"] not in existing_ids:
                docs.append(doc)

    # Generate the report
    prediction_with_defer = {**prediction, "deferred_to_human": state["uncertainty"].get("defer", False)}
    result = pipeline.context.report_fn(prediction_with_defer, docs)
    
    # Cache generated report to the database
    db = SessionLocal()
    try:
        db_assess = db.query(DBAssessment).filter(DBAssessment.session_id == session_id).first()
        if db_assess:
            db_assess.report_narrative = result.narrative
            db_assess.report_citations = json.dumps(result.citations)
            db_assess.report_generator = getattr(result, "generator", "template")
            db_assess.report_fallback_reason = getattr(result, "fallback_reason", None)
            db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error updating report in DB: {e}")
    finally:
        db.close()

    return ReportResponse(
        session_id=session_id,
        narrative=result.narrative,
        citations=result.citations,
        cached=result.cached,
        generator=getattr(result, "generator", "template"),
        fallback_reason=getattr(result, "fallback_reason", None),
    )


@app.post("/follow-up", response_model=FollowUpResponse)
def follow_up(request: FollowUpRequest, current_user: dict = Depends(get_current_user)) -> FollowUpResponse:
    state = _get_session_or_reconstruct(request.session_id)
    result = answer_follow_up(state, request.question, pipeline.context)
    return FollowUpResponse(
        session_id=request.session_id,
        answer=result.narrative,
        citations=result.citations,
        cached=result.cached,
        generator=getattr(result, "generator", "template"),
        fallback_reason=getattr(result, "fallback_reason", None),
    )
