import pytest

from src.reasoning.retriever import ClinicalKBRetriever, KBDocument, load_knowledge_base


def test_load_knowledge_base_from_real_data_dir():
    documents = load_knowledge_base()
    assert len(documents) > 10
    assert all(isinstance(d, KBDocument) for d in documents)
    assert all(d.source and d.text and d.category for d in documents)


def test_load_knowledge_base_raises_clear_error_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError, match="knowledge-base"):
        load_knowledge_base(tmp_path)


@pytest.fixture
def sample_documents():
    return [
        KBDocument(id="a", text="Sleep quality and mood are closely linked in mental health screening.", source="doc-a", category="test"),
        KBDocument(id="b", text="Heart rate variability is a physiological marker of stress arousal.", source="doc-b", category="test"),
        KBDocument(id="c", text="Crisis hotlines provide 24/7 support for people in emotional distress.", source="doc-c", category="test"),
    ]


def test_retriever_falls_back_to_tfidf_when_sentence_transformers_unavailable(sample_documents):
    # HF Hub is not reachable in this sandbox, so "auto" must fall back cleanly.
    retriever = ClinicalKBRetriever(embedding_backend="auto")
    retriever.build_index(sample_documents)
    assert retriever._active_backend == "tfidf"


def test_retriever_returns_relevant_documents_with_citations(sample_documents):
    retriever = ClinicalKBRetriever(embedding_backend="tfidf")
    retriever.build_index(sample_documents)

    results = retriever.retrieve("sleep and mood", k=2)
    assert len(results) <= 2
    assert results[0]["id"] == "a"  # most lexically relevant to the query
    for r in results:
        assert set(r.keys()) == {"id", "text", "source", "category", "score"}
        assert r["source"]  # every result must carry a citation


def test_retriever_raises_before_build_index():
    retriever = ClinicalKBRetriever(embedding_backend="tfidf")
    with pytest.raises(RuntimeError):
        retriever.retrieve("anything")


def test_retriever_respects_k():
    docs = [
        KBDocument(id=str(i), text=f"placeholder clinical text number {i} about stress", source=f"s{i}", category="test")
        for i in range(10)
    ]
    retriever = ClinicalKBRetriever(embedding_backend="tfidf")
    retriever.build_index(docs)
    results = retriever.retrieve("stress", k=3)
    assert len(results) == 3
