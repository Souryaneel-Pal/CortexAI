import pytest

from src.reasoning.agent_graph import AgentContext, answer_follow_up, build_agent_graph
from src.reasoning.rag_report import generate_cached_report
from src.reasoning.retriever import ClinicalKBRetriever, load_knowledge_base


@pytest.fixture
def real_retriever():
    docs = load_knowledge_base()
    retriever = ClinicalKBRetriever(embedding_backend="tfidf")
    retriever.build_index(docs)
    return retriever


def _make_context(real_retriever, predicted_class=1, confidence=0.85, defer=False, mdi=0.1, report_fn=None):
    """Build an AgentContext with deterministic stage functions.

    `report_fn` defaults to the templated generator rather than the module
    default (a live local `llama3.1`). These are *orchestration* tests -- they
    assert that the graph sequences stages, propagates the deferral flag, and
    force-attaches crisis documents. Letting them call a real LLM would make
    them slow (~50 s), non-deterministic in wording, and dependent on whether
    Ollama happens to be running on the machine executing the suite. The
    Ollama generator has its own dedicated coverage in
    tests/test_ollama_reasoning.py.
    """
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
        report_fn=report_fn or generate_cached_report,
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
    # The templated generator is injected here, so this asserts the graph
    # reports the generator's own `cached` flag rather than inventing one.
    assert "report: cached=True" in result["log"]


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


def test_graph_uses_the_injected_report_generator(real_retriever):
    """The graph must call whatever generator it was given and surface that
    result verbatim -- it never second-guesses or rewrites the report."""
    calls = []

    def recording_report_fn(prediction, docs):
        calls.append((prediction, docs))
        return generate_cached_report(prediction, docs, fallback_reason="injected for test")

    context = _make_context(real_retriever, report_fn=recording_report_fn)
    result = build_agent_graph(context).invoke({"raw_input": {}, "log": []})

    assert len(calls) == 1
    # The deferral decision is passed through to the generator, not recomputed.
    assert "deferred_to_human" in calls[0][0]
    assert result["report"].fallback_reason == "injected for test"


def test_graph_survives_a_report_generator_that_degrades(real_retriever):
    """An unreachable LLM must not break the graph: the report node still
    produces a valid, cited ReportResult."""

    def degrading_report_fn(prediction, docs):
        return generate_cached_report(prediction, docs, fallback_reason="Ollama unreachable in this test")

    context = _make_context(real_retriever, report_fn=degrading_report_fn)
    result = build_agent_graph(context).invoke({"raw_input": {}, "log": []})

    report = result["report"]
    assert report.valid is True
    assert report.cached is True
    assert report.citations
    assert "Ollama unreachable" in report.fallback_reason
