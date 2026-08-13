"""CortexAI FastAPI backend (docs/PROJECT_PLAN.md P5).

Endpoints: POST /assess (predict), GET /explain/{session_id},
GET /report/{session_id}, POST /follow-up, GET /health.

Session state (the AgentState produced by src/reasoning/agent_graph.py) is
kept in a simple in-memory dict keyed by session_id so /explain, /report,
and /follow-up can reuse a prior /assess call without recomputing the
prediction. This is an intentionally minimal MVP store (docs/MINDSCOPE_Blueprint.pdf's
full tech-stack table names Redis/Postgres for this in production) -- swap
`SESSION_STORE` for a real backing store before any multi-instance deploy.
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.inference import CortexAIPipeline
from src.api.schemas import (
    AssessmentRequest,
    CounterfactualResponse,
    ExplanationResponse,
    FollowUpRequest,
    FollowUpResponse,
    ModalityWeights,
    PredictionResponse,
    ReportResponse,
    ScoreBreakdown,
    UncertaintyInfo,
)
from src.data.schemas import StressLevel
from src.explain.counterfactual import format_counterfactual_narrative, generate_counterfactual_grid_search
from src.reasoning.agent_graph import answer_follow_up, build_agent_graph

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

pipeline: CortexAIPipeline | None = None
agent_app = None
SESSION_STORE: dict[str, dict] = {}


@app.on_event("startup")
def _startup() -> None:
    global pipeline, agent_app
    pipeline = CortexAIPipeline()
    agent_app = build_agent_graph(pipeline.context)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "is_demo_untrained_model": pipeline.is_demo_untrained_model if pipeline else None,
    }


@app.post("/assess", response_model=PredictionResponse)
def assess(request: AssessmentRequest) -> PredictionResponse:
    raw_input = {
        "tabular_features": request.tabular_features.to_ordered_list(),
        "face_image_base64": request.face_image_base64,
        "speech_audio_base64": request.speech_audio_base64,
    }

    session_id = str(uuid.uuid4())
    final_state = agent_app.invoke({"raw_input": raw_input, "log": []})
    SESSION_STORE[session_id] = final_state

    prediction = final_state["prediction"]
    uncertainty = final_state["uncertainty"]

    return PredictionResponse(
        session_id=session_id,
        predicted_class=pipeline.predicted_class_name(prediction["predicted_class"]),
        confidence=prediction["confidence"],
        scores=ScoreBreakdown(**prediction["scores"]),
        modality_weights=ModalityWeights(**prediction["modality_weights"]),
        uncertainty=UncertaintyInfo(**uncertainty),
        deferred_to_human=uncertainty["defer"],
        is_demo_untrained_model=pipeline.is_demo_untrained_model,
    )


def _get_session(session_id: str) -> dict:
    state = SESSION_STORE.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown session_id -- call /assess first.")
    return state


@app.get("/explain/{session_id}", response_model=ExplanationResponse)
def explain(session_id: str) -> ExplanationResponse:
    state = _get_session(session_id)
    explanations = state["explanations"]
    return ExplanationResponse(
        session_id=session_id,
        top_shap_features=explanations["shap_ranked_features"][:5],
        modality_weights=ModalityWeights(**state["prediction"]["modality_weights"]),
        masked_distress_index=explanations.get("masked_distress_index"),
    )


@app.get("/counterfactual/{session_id}", response_model=CounterfactualResponse)
def counterfactual(session_id: str, target_class: int | None = None) -> CounterfactualResponse:
    state = _get_session(session_id)
    prediction = state["prediction"]
    current_class = prediction["predicted_class"]

    if target_class is None:
        target_class = max(int(current_class) - 1, int(StressLevel.HEALTHY))
    if target_class == current_class:
        return CounterfactualResponse(session_id=session_id, counterfactual=None, narrative=None)

    from src.data.schemas import TABULAR_FEATURE_COLUMNS

    tabular = state["preprocessed"]["tabular"][0].tolist()
    query = dict(zip(TABULAR_FEATURE_COLUMNS, tabular))
    # Actionable, behavioural levers only -- not physiological readings a
    # person can't directly control (docs/Proposal.pdf Sec. 06/08 differentiator #3).
    actionable_features = ["Sleep_Quality", "Social_Engagement", "Daily_App_Usage_Min", "Session_Frequency"]
    ranges = {f: (0.0, 5.0) if f == "Sleep_Quality" or f == "Social_Engagement" else (0.0, 300.0) for f in actionable_features}

    result = generate_counterfactual_grid_search(
        pipeline.tabular_encoder,
        query_instance=query,
        feature_ranges=ranges,
        desired_class=target_class,
        features_to_vary=actionable_features,
    )
    narrative = format_counterfactual_narrative(result) if result else None
    return CounterfactualResponse(session_id=session_id, counterfactual=result, narrative=narrative)


@app.get("/report/{session_id}", response_model=ReportResponse)
def report(session_id: str) -> ReportResponse:
    state = _get_session(session_id)
    result = state["report"]
    return ReportResponse(session_id=session_id, narrative=result.narrative, citations=result.citations, cached=result.cached)


@app.post("/follow-up", response_model=FollowUpResponse)
def follow_up(request: FollowUpRequest) -> FollowUpResponse:
    state = _get_session(request.session_id)
    result = answer_follow_up(state, request.question, pipeline.context)
    return FollowUpResponse(
        session_id=request.session_id, answer=result.narrative, citations=result.citations, cached=result.cached
    )
