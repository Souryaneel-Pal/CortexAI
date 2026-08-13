"""Thin agent orchestrator (docs/PROJECT_PLAN.md P4; docs/MINDSCOPE_Blueprint.pdf
Sec. 05 Layer 4: "A thin conductor: runs encoders, checks the uncertainty
gate, decides when to defer to a human, assembles the report, logs
everything. It coordinates -- it never diagnoses.")

A LangGraph state machine sequencing:
    preprocess -> predict -> uncertainty_gate -> explain -> retrieve -> report

Every node is a thin wrapper around functions injected via `AgentContext` --
the real preprocessing/model/explainability/retrieval/report code (P1-P4)
lives elsewhere; this module only sequences it and enforces two invariants
regardless of what the injected functions do:
  1. It never overrides or second-guesses the prediction -- it only routes
     around it (gates on uncertainty, decides what to explain/retrieve).
  2. Crisis-resource documents are always attached when severity or the
     Masked-Distress Index is high, never left to retrieval-ranking chance.

`answer_follow_up` handles interactive questions ("why moderate?") after a
full run, grounded in the same prediction/explanations/retrieved_docs the
main run already computed -- it never re-predicts, only re-explains.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypedDict

from src.data.schemas import StressLevel
from src.explain.masked_distress import DEFAULT_MDI_FLAG_THRESHOLD
from src.reasoning.rag_report import ReportResult, generate_report
from src.reasoning.retriever import ClinicalKBRetriever

CRISIS_TRIGGER_CLASS = int(StressLevel.SEVERE_STRESS)


class AgentState(TypedDict, total=False):
    raw_input: dict
    preprocessed: dict
    prediction: dict
    uncertainty: dict
    explanations: dict
    retrieved_docs: list[dict]
    report: Any  # ReportResult
    log: list[str]


@dataclass
class AgentContext:
    """Injected implementations for each pipeline stage. Kept separate from
    the graph so the graph itself has no dependency on which encoders/
    explainers/retriever instance is in use -- src/api/inference.py (P5)
    constructs the real one; tests construct a fake one.
    """

    preprocess_fn: Callable[[dict], dict]
    predict_fn: Callable[[dict], dict]
    uncertainty_fn: Callable[[dict], dict]
    explain_fn: Callable[[dict, dict], dict]
    retriever: ClinicalKBRetriever
    report_fn: Callable[[dict, list[dict]], ReportResult] = field(default=generate_report)
    top_k_docs: int = 4


def _log(state: AgentState, message: str) -> list[str]:
    return [*state.get("log", []), message]


def build_agent_graph(context: AgentContext):
    """Returns a compiled LangGraph app with a single `.invoke(state)` entrypoint."""
    from langgraph.graph import END, StateGraph

    def preprocess_node(state: AgentState) -> dict:
        preprocessed = context.preprocess_fn(state["raw_input"])
        return {"preprocessed": preprocessed, "log": _log(state, "preprocess: ok")}

    def predict_node(state: AgentState) -> dict:
        prediction = context.predict_fn(state["preprocessed"])
        return {"prediction": prediction, "log": _log(state, f"predict: class={prediction.get('predicted_class')}")}

    def uncertainty_node(state: AgentState) -> dict:
        uncertainty = context.uncertainty_fn(state["prediction"])
        return {"uncertainty": uncertainty, "log": _log(state, f"uncertainty_gate: defer={uncertainty.get('defer')}")}

    def explain_node(state: AgentState) -> dict:
        explanations = context.explain_fn(state["preprocessed"], state["prediction"])
        return {"explanations": explanations, "log": _log(state, "explain: ok")}

    def retrieve_node(state: AgentState) -> dict:
        prediction = state["prediction"]
        explanations = state.get("explanations", {})

        query_parts = [f"stress status {prediction.get('predicted_class')}"]
        top_feature = explanations.get("top_shap_feature")
        if top_feature:
            query_parts.append(str(top_feature))
        query = " ".join(query_parts)

        docs = context.retriever.retrieve(query, k=context.top_k_docs)

        # Crisis resources are never left to similarity-search chance: always
        # attach them when severity is high or the Masked-Distress Index
        # flags a contradiction, regardless of whether they ranked in the
        # top-k for the free-text query above.
        mdi = explanations.get("masked_distress_index", {}).get("mdi", 0.0)
        severe = prediction.get("predicted_class") == CRISIS_TRIGGER_CLASS
        masked_distress_flagged = mdi >= DEFAULT_MDI_FLAG_THRESHOLD
        if severe or masked_distress_flagged:
            existing_ids = {d["id"] for d in docs}
            for doc in context.retriever.get_by_category("crisis_resource"):
                if doc["id"] not in existing_ids:
                    docs.append(doc)

        reason = "severe_class" if severe else "masked_distress" if masked_distress_flagged else None
        return {
            "retrieved_docs": docs,
            "log": _log(state, f"retrieve: {len(docs)} docs" + (f" (crisis docs added: {reason})" if reason else "")),
        }

    def report_node(state: AgentState) -> dict:
        prediction = {**state["prediction"], "deferred_to_human": state["uncertainty"].get("defer", False)}
        report = context.report_fn(prediction, state["retrieved_docs"])
        return {"report": report, "log": _log(state, f"report: cached={getattr(report, 'cached', None)}")}

    graph = StateGraph(AgentState)
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("predict", predict_node)
    graph.add_node("uncertainty_gate", uncertainty_node)
    graph.add_node("explain", explain_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "predict")
    graph.add_edge("predict", "uncertainty_gate")
    graph.add_edge("uncertainty_gate", "explain")
    graph.add_edge("explain", "retrieve")
    graph.add_edge("retrieve", "report")
    graph.add_edge("report", END)

    return graph.compile()


def answer_follow_up(
    final_state: AgentState,
    question: str,
    context: AgentContext,
) -> ReportResult:
    """Answers an interactive follow-up ("why moderate?") using the SAME
    prediction and explanations already computed by the main run -- never
    re-predicts, only re-explains and re-grounds. Retrieves a few more docs
    targeted at the question text and asks the report generator for a
    focused, cited answer.
    """
    extra_docs = context.retriever.retrieve(question, k=context.top_k_docs)
    existing_ids = {d["id"] for d in final_state.get("retrieved_docs", [])}
    combined_docs = [*final_state.get("retrieved_docs", []), *[d for d in extra_docs if d["id"] not in existing_ids]]

    prediction = {
        **final_state["prediction"],
        "deferred_to_human": final_state.get("uncertainty", {}).get("defer", False),
        "follow_up_question": question,
    }
    return context.report_fn(prediction, combined_docs)
