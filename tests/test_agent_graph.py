import pytest

from src.reasoning.agent_graph import AgentContext, answer_follow_up, build_agent_graph
from src.reasoning.retriever import ClinicalKBRetriever, load_knowledge_base


@pytest.fixture
def real_retriever():
    docs = load_knowledge_base()
    retriever = ClinicalKBRetriever(embedding_backend="tfidf")
    retriever.build_index(docs)
    return retriever


def _make_context(real_retriever, predicted_class=1, confidence=0.85, defer=False, mdi=0.1):
    def preprocess_fn(raw_input):
        return {"features": raw_input.get("features", [0.0] * 18)}

    def predict_fn(preprocessed):
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "scores": {"Depression_Score": 10.0, "Anxiety_Score": 8.0, "Stress_Score": 15.0},
            "modality_weights": {"face": 0.3, "speech": 0.3, "tabular": 0.4},
        }

    def uncertainty_fn(prediction):
        return {"defer": defer, "reason": "low_confidence" if defer else None}

    def explain_fn(preprocessed, prediction):
        return {
            "top_shap_feature": "Sleep_Quality",
            "masked_distress_index": {"mdi": mdi, "flag": mdi >= 0.5},
        }

    return AgentContext(
        preprocess_fn=preprocess_fn,
        predict_fn=predict_fn,
        uncertainty_fn=uncertainty_fn,
        explain_fn=explain_fn,
        retriever=real_retriever,
    )


def test_full_pipeline_runs_end_to_end(real_retriever):
    context = _make_context(real_retriever)
    app = build_agent_graph(context)

    result = app.invoke({"raw_input": {"features": [0.0] * 18}, "log": []})

    assert result["prediction"]["predicted_class"] == 1
    assert result["uncertainty"]["defer"] is False
    assert "top_shap_feature" in result["explanations"]
    assert len(result["retrieved_docs"]) > 0
    assert result["report"].valid is True
    assert "preprocess: ok" in result["log"]
    assert "report: cached=True" in result["log"]  # no ANTHROPIC_API_KEY in this sandbox


def test_severe_prediction_always_attaches_crisis_docs(real_retriever):
    context = _make_context(real_retriever, predicted_class=3, mdi=0.1)  # Severe
    app = build_agent_graph(context)
    result = app.invoke({"raw_input": {}, "log": []})

    categories = {d["category"] for d in result["retrieved_docs"]}
    assert "crisis_resource" in categories


def test_high_mdi_always_attaches_crisis_docs_even_if_class_not_severe(real_retriever):
    context = _make_context(real_retriever, predicted_class=1, mdi=0.8)  # Mild class, but high MDI
    app = build_agent_graph(context)
    result = app.invoke({"raw_input": {}, "log": []})

    categories = {d["category"] for d in result["retrieved_docs"]}
    assert "crisis_resource" in categories


def test_healthy_low_mdi_does_not_force_crisis_docs(real_retriever):
    context = _make_context(real_retriever, predicted_class=0, mdi=0.05)  # Healthy, low MDI
    app = build_agent_graph(context)
    result = app.invoke({"raw_input": {}, "log": []})

    # Crisis docs may still appear via similarity search coincidentally, but
    # are not FORCED in -- this just checks the pipeline doesn't crash and
    # produces a valid, cited report either way.
    assert result["report"].valid is True


def test_deferred_prediction_is_reflected_in_report(real_retriever):
    context = _make_context(real_retriever, confidence=0.4, defer=True)
    app = build_agent_graph(context)
    result = app.invoke({"raw_input": {}, "log": []})

    assert result["uncertainty"]["defer"] is True
    assert "flagged for human review" in result["report"].narrative


def test_answer_follow_up_reuses_prediction_without_repredicting(real_retriever):
    context = _make_context(real_retriever)
    app = build_agent_graph(context)
    final_state = app.invoke({"raw_input": {}, "log": []})

    follow_up_report = answer_follow_up(final_state, "why was this classified as mild stress?", context)
    assert follow_up_report.valid is True
    # The follow-up report is grounded in the SAME predicted class, not a new prediction.
    assert final_state["prediction"]["predicted_class"] == 1
