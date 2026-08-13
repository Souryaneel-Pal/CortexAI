"""Clinical knowledge-base retrieval (docs/PROJECT_PLAN.md P4;
docs/MINDSCOPE_Blueprint.pdf Sec. 05 Layer 3 "Grounding": "Retrieves from a
curated clinical knowledge base ... and writes a citation-backed narrative.
Every clinical claim traces to a source -- no hallucinated advice.")

Primary embedding backend: sentence-transformers + FAISS (docs/MINDSCOPE_Blueprint.pdf
Sec. 08). Automatic fallback to TF-IDF + cosine similarity (scikit-learn,
already a hard dependency, no model download) when the sentence-transformers
model can't be downloaded -- verified in this sandbox: the HuggingFace Hub is
not reachable from here (same restriction hit by the wav2vec2 speech
encoder), so the fallback path is what's actually exercised and tested here;
the primary path activates automatically wherever HF Hub access exists.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"


@dataclass
class KBDocument:
    id: str
    text: str
    source: str
    category: str


def load_knowledge_base(directory: Path | str = KNOWLEDGE_BASE_DIR) -> list[KBDocument]:
    """Loads every *.json file in the knowledge base directory. Each file is
    a list of {"id", "text", "source", "category"} objects -- see
    data/knowledge_base/README.md for the placeholder-content disclosure
    every entry in this directory carries.
    """
    directory = Path(directory)
    documents: list[KBDocument] = []
    for path in sorted(directory.glob("*.json")):
        entries = json.loads(path.read_text())
        for entry in entries:
            documents.append(KBDocument(**entry))
    if not documents:
        raise FileNotFoundError(
            f"No knowledge-base *.json files found under {directory}. "
            f"The RAG layer has nothing to ground its citations in."
        )
    return documents


class ClinicalKBRetriever:
    """Embeds and indexes `KBDocument`s, retrieves the top-k most relevant
    documents for a query -- the only source of text the report generator
    (src/reasoning/rag_report.py) is allowed to cite.
    """

    def __init__(self, embedding_backend: str = "auto", model_name: str = "all-MiniLM-L6-v2"):
        self.documents: list[KBDocument] = []
        self.embedding_backend = embedding_backend
        self.model_name = model_name
        self._encoder = None  # sentence-transformers model, if that backend is active
        self._vectorizer = None  # sklearn TfidfVectorizer, if that backend is active
        self._doc_matrix = None
        self._faiss_index = None
        self._active_backend: str | None = None

    def _init_sentence_transformer_backend(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
            self._active_backend = "sentence_transformers"
            return True
        except Exception as exc:  # network/model-download failure, or lib not installed
            logger.warning(
                f"sentence-transformers backend unavailable ({type(exc).__name__}: {exc}); "
                f"falling back to TF-IDF retrieval. Citations will still be exact and "
                f"sourced -- only the relevance ranking quality is reduced."
            )
            return False

    def build_index(self, documents: list[KBDocument]) -> None:
        self.documents = documents
        texts = [doc.text for doc in documents]

        if self.embedding_backend in ("auto", "sentence_transformers"):
            if self._init_sentence_transformer_backend():
                import faiss

                embeddings = self._encoder.encode(texts, normalize_embeddings=True)
                embeddings = np.asarray(embeddings, dtype=np.float32)
                self._faiss_index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine sim via normalized inner product
                self._faiss_index.add(embeddings)
                return

        # TF-IDF fallback.
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._doc_matrix = self._vectorizer.fit_transform(texts)
        self._active_backend = "tfidf"

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Returns up to `k` documents as
        [{"id", "text", "source", "category", "score"}, ...], sorted by
        descending relevance score.
        """
        if not self.documents:
            raise RuntimeError("build_index() must be called before retrieve().")

        if self._active_backend == "sentence_transformers":
            query_embedding = np.asarray(
                self._encoder.encode([query], normalize_embeddings=True), dtype=np.float32
            )
            scores, indices = self._faiss_index.search(query_embedding, min(k, len(self.documents)))
            scores, indices = scores[0], indices[0]
        else:
            from sklearn.metrics.pairwise import cosine_similarity

            query_vector = self._vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self._doc_matrix)[0]
            top_k = np.argsort(-similarities)[:k]
            scores, indices = similarities[top_k], top_k

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            doc = self.documents[int(idx)]
            results.append(
                {"id": doc.id, "text": doc.text, "source": doc.source, "category": doc.category, "score": float(score)}
            )
        return results

    def get_by_category(self, category: str) -> list[dict]:
        """Direct category lookup, bypassing similarity ranking -- used to
        guarantee crisis-resource documents are always surfaced when
        warranted (src/reasoning/agent_graph.py), rather than leaving that
        safety-critical inclusion to a similarity-search top-k that might
        not rank them highly enough to appear.
        """
        return [
            {"id": d.id, "text": d.text, "source": d.source, "category": d.category, "score": None}
            for d in self.documents
            if d.category == category
        ]
