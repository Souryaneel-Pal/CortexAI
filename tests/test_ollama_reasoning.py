"""Tests for the local-Ollama RAG stack (src/reasoning/ollama_config.py,
retriever.py, rag_report.py).

These deliberately do NOT require a running Ollama. The whole point of this
layer is that it behaves correctly when Ollama is *absent*, so the failure
paths are forced explicitly (bad base URL, unknown model, exploding client)
rather than depending on what happens to be installed on the test machine.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.reasoning.ollama_config import (
    OllamaStatus,
    embedding_model,
    llm_model,
    probe_ollama,
    reset_probe_cache,
)
from src.reasoning.rag_report import (
    ReportResult,
    contains_required_framing,
    generate_report_ollama,
    validate_citations,
)
from src.reasoning.retriever import ClinicalKBRetriever, KBDocument, _NumpyFlatIPIndex


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    """The probe result is cached process-wide; don't leak it between tests."""
    reset_probe_cache()
    yield
    reset_probe_cache()


@pytest.fixture
def sample_documents():
    return [
        KBDocument(id="a", text="Sleep quality and mood are closely linked.", source="doc-a", category="score_band"),
        KBDocument(id="b", text="Heart rate variability marks stress arousal.", source="doc-b", category="symptom_domain"),
        KBDocument(id="c", text="Crisis hotlines provide 24/7 support.", source="doc-c", category="crisis_resource"),
    ]


@pytest.fixture
def prediction_result():
    return {
        "predicted_class": 3,
        "confidence": 0.72,
        "scores": {"Depression_Score": 28.4, "Anxiety_Score": 17.1, "Stress_Score": 31.2},
        "modality_weights": {"face": 0.5, "speech": 0.3, "tabular": 0.2},
        "deferred_to_human": False,
        "masked_distress_index": 0.81,
    }


# ---------------------------------------------------------------------------
# Health probing
# ---------------------------------------------------------------------------
def test_probe_reports_unreachable_without_raising(monkeypatch):
    """An offline Ollama is a normal operating condition, not an exception."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:59999")
    reset_probe_cache()

    status = probe_ollama(force=True)
    assert status.reachable is False
    assert status.error
    assert status.has_model("llama3.1") is False

    reason = status.unavailable_reason("llama3.1")
    assert "127.0.0.1:59999" in reason
    assert "ollama serve" in reason  # the message must say how to fix it


def test_status_matches_models_with_and_without_latest_tag():
    """Ollama reports `llama3.1:latest`; a bare name must still match."""
    status = OllamaStatus(reachable=True, base_url="http://x", models=["llama3.1:latest", "nomic-embed-text:latest"])
    assert status.has_model("llama3.1")
    assert status.has_model("llama3.1:latest")
    assert status.has_model("nomic-embed-text")
    assert not status.has_model("mistral")


def test_unavailable_reason_names_the_pull_command():
    status = OllamaStatus(reachable=True, base_url="http://x", models=["phi3:mini"])
    reason = status.unavailable_reason("llama3.1")
    assert "ollama pull llama3.1" in reason
    assert "phi3:mini" in reason  # and lists what IS available


def test_configured_model_names_default_to_the_documented_ones(monkeypatch):
    monkeypatch.delenv("CORTEXAI_OLLAMA_LLM_MODEL", raising=False)
    monkeypatch.delenv("CORTEXAI_OLLAMA_EMBED_MODEL", raising=False)
    assert llm_model() == "llama3.1"
    assert embedding_model() == "nomic-embed-text"


# ---------------------------------------------------------------------------
# Report generation fallbacks
# ---------------------------------------------------------------------------
def test_report_falls_back_cleanly_when_ollama_unreachable(monkeypatch, sample_documents, prediction_result):
    """The headline requirement: Ollama offline must produce a clear,
    user-friendly response through the API -- not an unhandled exception."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:59999")
    reset_probe_cache()

    docs = [{"id": d.id, "text": d.text, "source": d.source, "category": d.category, "score": 1.0} for d in sample_documents]
    result = generate_report_ollama(prediction_result, docs)

    assert isinstance(result, ReportResult)
    assert result.cached is True
    assert result.generator == "template"
    assert result.valid is True
    assert "ollama serve" in result.fallback_reason
    # The degradation is disclosed inside the narrative the clinician reads.
    assert "unavailable" in result.narrative.lower()
    # And the report is still correctly framed and fully cited.
    assert contains_required_framing(result.narrative)
    assert validate_citations(result.narrative, docs)[0]


def test_report_falls_back_when_model_not_pulled(monkeypatch, sample_documents, prediction_result):
    monkeypatch.setattr(
        "src.reasoning.rag_report.probe_ollama",
        lambda: OllamaStatus(reachable=True, base_url="http://localhost:11434", models=["phi3:mini"]),
    )
    docs = [{"id": d.id, "text": d.text, "source": d.source, "category": d.category, "score": 1.0} for d in sample_documents]

    result = generate_report_ollama(prediction_result, docs, model="llama3.1")
    assert result.cached is True
    assert "ollama pull llama3.1" in result.fallback_reason


def test_report_falls_back_when_generation_raises(sample_documents, prediction_result):
    """A connection reset or timeout mid-generation must degrade, not 500."""

    class ExplodingChat:
        def invoke(self, _messages):
            raise ConnectionError("connection reset by peer")

    docs = [{"id": d.id, "text": d.text, "source": d.source, "category": d.category, "score": 1.0} for d in sample_documents]
    result = generate_report_ollama(prediction_result, docs, chat_model=ExplodingChat())

    assert result.cached is True
    assert result.valid is True
    assert "ConnectionError" in result.fallback_reason


def test_report_rejects_hallucinated_citations(sample_documents, prediction_result):
    """A model citing a source that wasn't retrieved must have its narrative
    discarded, not merely flagged -- this is the layer's core guarantee."""

    class HallucinatingChat:
        def invoke(self, _messages):
            class Response:
                content = "Stress is elevated [totally-made-up-doc] and sleep is poor [a]."

            return Response()

    docs = [{"id": d.id, "text": d.text, "source": d.source, "category": d.category, "score": 1.0} for d in sample_documents]
    result = generate_report_ollama(prediction_result, docs, chat_model=HallucinatingChat())

    assert result.cached is True, "hallucinated citation must fall back to the templated report"
    assert "totally-made-up-doc" in result.fallback_reason
    assert result.invalid_citations == ["totally-made-up-doc"]

    # The hallucinated *claim* is gone and the bad id survives only inside the
    # disclosure line naming what went wrong -- never as a citation marker.
    assert "[totally-made-up-doc]" not in result.narrative
    assert "Stress is elevated" not in result.narrative
    assert validate_citations(result.narrative, docs)[0]
    assert "totally-made-up-doc" in result.narrative  # disclosed, not hidden


def test_successful_generation_is_labelled_as_live(sample_documents, prediction_result):
    class GoodChat:
        def invoke(self, _messages):
            class Response:
                content = "This is decision-support information, not a diagnosis. Sleep quality matters [a]."

            return Response()

    docs = [{"id": d.id, "text": d.text, "source": d.source, "category": d.category, "score": 1.0} for d in sample_documents]
    result = generate_report_ollama(prediction_result, docs, chat_model=GoodChat(), model="llama3.1")

    assert result.cached is False
    assert result.generator == "ollama:llama3.1"
    assert result.fallback_reason is None
    assert result.citations == ["a"]


# ---------------------------------------------------------------------------
# Vector index
# ---------------------------------------------------------------------------
# Importing faiss into a process that already has PyTorch loaded aborts the
# interpreter (duplicate OpenMP runtimes -- see
# ClinicalKBRetriever._faiss_is_safe_here). Most of the test session has torch
# loaded, so the FAISS comparison runs in a clean subprocess. Skipping it
# instead would leave the "exact substitute" claim unverified, which is the
# one property the NumPy index has to earn.
_FAISS_EQUIVALENCE_SUBPROCESS = """
import faiss
import numpy as np

from src.reasoning.retriever import _NumpyFlatIPIndex

rng = np.random.default_rng(0)
docs = rng.random((40, 64)).astype(np.float32)
docs /= np.linalg.norm(docs, axis=1, keepdims=True)
queries = rng.random((3, 64)).astype(np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

reference = faiss.IndexFlatIP(64)
reference.add(docs)
expected_scores, expected_idx = reference.search(queries, 5)

actual_scores, actual_idx = _NumpyFlatIPIndex(docs).search(queries, 5)

assert np.array_equal(actual_idx, expected_idx), (actual_idx, expected_idx)
assert np.allclose(actual_scores, expected_scores, atol=1e-5)
print("OK")
"""


def test_numpy_index_matches_faiss_exactly():
    """The NumPy index is used whenever FAISS can't load beside PyTorch, so it
    has to be an exact substitute for `IndexFlatIP`, not an approximation."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", _FAISS_EQUIVALENCE_SUBPROCESS],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"FAISS equivalence subprocess failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout


def test_numpy_index_orders_results_by_descending_score():
    embeddings = np.eye(4, dtype=np.float32)
    index = _NumpyFlatIPIndex(embeddings)
    scores, indices = index.search(np.array([[0.1, 0.9, 0.4, 0.2]], dtype=np.float32), k=3)
    assert list(indices[0]) == [1, 2, 3]
    assert list(scores[0]) == sorted(scores[0], reverse=True)


def test_retriever_degrades_to_tfidf_when_ollama_offline(monkeypatch, sample_documents):
    """With Ollama down AND sentence-transformers forced to fail, retrieval
    must still work -- citations are exact in every backend."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:59999")
    reset_probe_cache()
    monkeypatch.setattr(
        ClinicalKBRetriever, "_init_sentence_transformer_backend", lambda self: False, raising=True
    )

    retriever = ClinicalKBRetriever(embedding_backend="auto")
    retriever.build_index(sample_documents)

    assert retriever.active_backend == "tfidf"
    assert "ollama serve" in retriever.degraded_reason
    results = retriever.retrieve("sleep and mood", k=2)
    assert results and all(r["source"] for r in results)


def test_explicit_ollama_backend_raises_rather_than_silently_degrading(monkeypatch, sample_documents):
    """`embedding_backend='auto'` degrades; an explicit 'ollama' request must
    fail loudly instead, so a deployment that requires local embeddings finds
    out at startup rather than serving lower-quality rankings unnoticed."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:59999")
    reset_probe_cache()

    retriever = ClinicalKBRetriever(embedding_backend="ollama")
    with pytest.raises(RuntimeError, match="ollama"):
        retriever.build_index(sample_documents)


def test_uncited_narrative_is_rejected(sample_documents, prediction_result):
    """`validate_citations` is trivially true for a narrative with no citations
    at all, so acceptance must additionally require that at least one exists --
    otherwise a fluent but entirely unsourced narrative ships as 'grounded'.
    Observed with llama3.1 in practice, which is why this test exists."""

    class UncitedChat:
        def invoke(self, _messages):
            class Response:
                content = (
                    "This is decision-support information, not a diagnosis. The individual "
                    "shows elevated stress and reduced heart-rate variability."
                )

            return Response()

    docs = [{"id": d.id, "text": d.text, "source": d.source, "category": d.category, "score": 1.0} for d in sample_documents]
    result = generate_report_ollama(prediction_result, docs, chat_model=UncitedChat())

    assert result.cached is True
    assert "no citations" in result.fallback_reason
    assert result.citations, "the templated fallback must itself be cited"


def test_narrative_missing_responsible_ai_framing_is_rejected(sample_documents, prediction_result):
    """The decision-support framing is a safety requirement, not a stylistic
    preference -- a narrative that drops it must not reach the clinician."""

    class UnframedChat:
        def invoke(self, _messages):
            class Response:
                content = "The patient has severe stress and requires medication [a]."

            return Response()

    docs = [{"id": d.id, "text": d.text, "source": d.source, "category": d.category, "score": 1.0} for d in sample_documents]
    result = generate_report_ollama(prediction_result, docs, chat_model=UnframedChat())

    assert result.cached is True
    assert "framing" in result.fallback_reason
    assert contains_required_framing(result.narrative)
