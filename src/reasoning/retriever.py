"""Clinical knowledge-base retrieval (PROJECT_PLAN.md P4;
docs/MINDSCOPE_Blueprint.pdf Sec. 05 Layer 3 "Grounding": "Retrieves from a
curated clinical knowledge base ... and writes a citation-backed narrative.
Every clinical claim traces to a source -- no hallucinated advice.")

Embedding backends, tried in this order under `embedding_backend="auto"`:

  1. **ollama** -- `nomic-embed-text` served by a local Ollama instance,
     indexed with FAISS. This is the primary path: embeddings are computed
     on-device, so the clinical KB and every query stay local, which is the
     point of running this stack offline at all.
  2. **sentence_transformers** -- `all-MiniLM-L6-v2` + FAISS, for machines
     with no Ollama but with HuggingFace Hub access.
  3. **tfidf** -- scikit-learn, already a hard dependency, no model download
     and no network. Always available, so retrieval never hard-fails.

Every backend returns the same result shape and the same exact citations;
only the relevance *ranking quality* differs. That matters: a degraded
backend can rank a less-relevant guideline first, but it can never invent a
source, so the "no unsourced claim" guarantee holds regardless of which one
is active. `_active_backend` records which one actually ran, and the API
surfaces it so a degraded run is visible rather than silent.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.reasoning.ollama_config import embedding_model, probe_ollama

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"


@dataclass
class KBDocument:
    id: str
    text: str
    source: str
    category: str


class _NumpyFlatIPIndex:
    """Exact drop-in replacement for `faiss.IndexFlatIP`, same `.search()`
    signature and same return shapes `(scores, indices)`.

    `IndexFlatIP` performs brute-force exact inner-product search, so this
    is numerically equivalent rather than an approximation -- it exists only
    to avoid the FAISS/PyTorch OpenMP conflict documented in
    `ClinicalKBRetriever._faiss_is_safe_here`.
    """

    def __init__(self, embeddings: np.ndarray):
        self._embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self._embeddings.shape[0])

    def add(self, embeddings: np.ndarray) -> None:
        self._embeddings = np.vstack([self._embeddings, np.asarray(embeddings, dtype=np.float32)])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.float32)

        # `np.errstate` here suppresses *spurious* floating-point flags, not
        # real ones. NumPy 1.26 on macOS with the Accelerate BLAS backend
        # raises "divide by zero / overflow / invalid value encountered in
        # matmul" for inputs that are entirely finite and unit-normalised.
        # Verified on this machine: with clean float32 inputs the flags fire,
        # yet the result matches both np.einsum and an explicit Python
        # dot-product loop to 1.2e-07 (float32 epsilon) and is fully finite.
        # Leaving the warnings on would train readers to ignore genuine
        # numerical warnings from this file.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            similarities = queries @ self._embeddings.T  # (n_queries, n_docs)

        k = min(k, similarities.shape[1])
        # argpartition for the top-k, then sort just those -- same ordering
        # FAISS returns (descending score).
        top_unsorted = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
        ordered = np.take_along_axis(
            top_unsorted,
            np.argsort(-np.take_along_axis(similarities, top_unsorted, axis=1), axis=1),
            axis=1,
        )
        scores = np.take_along_axis(similarities, ordered, axis=1)
        return scores.astype(np.float32), ordered.astype(np.int64)


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

    def __init__(
        self,
        embedding_backend: str = "auto",
        model_name: str = "all-MiniLM-L6-v2",
        ollama_model: str | None = None,
    ):
        self.documents: list[KBDocument] = []
        self.embedding_backend = embedding_backend
        self.model_name = model_name
        self.ollama_model = ollama_model or embedding_model()
        self._encoder = None  # sentence-transformers model, if that backend is active
        self._ollama_embeddings = None  # langchain OllamaEmbeddings, if that backend is active
        self._vectorizer = None  # sklearn TfidfVectorizer, if that backend is active
        self._doc_matrix = None
        self._faiss_index = None
        self._active_backend: str | None = None
        self._degraded_reason: str | None = None
        self._index_impl: str | None = None

    @property
    def active_backend(self) -> str | None:
        """Which embedding backend actually ran. Surfaced by the API so a
        degraded retrieval run is visible instead of silent.
        """
        return self._active_backend

    @property
    def degraded_reason(self) -> str | None:
        """Why the preferred backend was skipped, or None if it wasn't."""
        return self._degraded_reason

    @property
    def index_impl(self) -> str | None:
        """Which vector index actually ran -- 'faiss.IndexFlatIP' or its exact
        NumPy equivalent (see `_faiss_is_safe_here`)."""
        return self._index_impl

    def _init_ollama_backend(self) -> bool:
        """`nomic-embed-text` via a local Ollama instance -- the primary path.

        Probes Ollama first so an offline instance costs one cheap, cached
        HTTP call rather than a per-document socket timeout while embedding
        the whole knowledge base.
        """
        status = probe_ollama()
        reason = status.unavailable_reason(self.ollama_model)
        if reason is not None:
            self._degraded_reason = reason
            logger.warning("Ollama embedding backend unavailable: %s", reason)
            return False

        try:
            try:
                from langchain_ollama import OllamaEmbeddings
            except ImportError:  # older split-package layout
                from langchain_community.embeddings import OllamaEmbeddings

            self._ollama_embeddings = OllamaEmbeddings(model=self.ollama_model, base_url=status.base_url)
            self._active_backend = "ollama"
            return True
        except Exception as exc:
            self._degraded_reason = f"Ollama embeddings failed to initialise ({type(exc).__name__}: {exc})"
            logger.warning(self._degraded_reason)
            return False

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

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)

    @staticmethod
    def _faiss_is_safe_here() -> bool:
        """Whether importing FAISS in *this* process is safe.

        FAISS (faiss-cpu on macOS/ARM) and PyTorch each link their own
        OpenMP runtime, and loading both aborts the interpreter outright:

            OMP: Error #15: Initializing libomp.dylib, but found
            libomp.dylib already initialized.

        It is an abort, not a Python exception, so it cannot be caught -- the
        decision has to be made *before* importing faiss. Both orders crash
        (verified faiss->torch and torch->faiss), so the only safe rule is:
        if torch is already in this process, don't import faiss.

        `KMP_DUPLICATE_LIB_OK=TRUE` does suppress the abort, and FAISS then
        returns results identical to the reference implementation -- but the
        same run also emits overflow/invalid-value warnings from unrelated
        numpy matmuls, which is exactly the "silently produce incorrect
        results" failure the OpenMP documentation warns about. Not a
        trade worth making in a system that scores mental-health severity.
        """
        import sys

        return "torch" not in sys.modules

    def _build_vector_index(self, embeddings: np.ndarray):
        """Cosine similarity via inner product over L2-normalised vectors.

        Uses `faiss.IndexFlatIP` where FAISS can safely load, and an exact
        NumPy equivalent otherwise. This is not an approximation swap:
        `IndexFlatIP` *is* brute-force exact inner-product search, so the
        NumPy path returns identical neighbours and identical scores (pinned
        by tests/test_retriever.py). With a 29-document clinical KB the two
        are also indistinguishable in latency.
        """
        embeddings = self._normalize(embeddings)

        if self._faiss_is_safe_here():
            import faiss

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            self._index_impl = "faiss.IndexFlatIP"
            return index

        logger.info(
            "PyTorch is loaded in this process, so FAISS cannot be imported safely "
            "(OpenMP runtime conflict). Using the exact NumPy inner-product equivalent "
            "of faiss.IndexFlatIP -- identical results, no approximation."
        )
        self._index_impl = "numpy.exact_inner_product"
        return _NumpyFlatIPIndex(embeddings)

    def build_index(self, documents: list[KBDocument]) -> None:
        self.documents = documents
        texts = [doc.text for doc in documents]

        if self.embedding_backend in ("auto", "ollama"):
            if self._init_ollama_backend():
                try:
                    embeddings = np.asarray(
                        self._ollama_embeddings.embed_documents(texts), dtype=np.float32
                    )
                    self._faiss_index = self._build_vector_index(embeddings)
                    logger.info(
                        "Retrieval backend: Ollama '%s' + FAISS (%d documents indexed)",
                        self.ollama_model,
                        len(texts),
                    )
                    return
                except Exception as exc:
                    # Ollama answered /api/tags but embedding failed (model
                    # pulled mid-probe, OOM, request timeout). Don't let the
                    # KB index fail outright -- degrade to the next backend.
                    self._degraded_reason = f"Ollama embedding call failed ({type(exc).__name__}: {exc})"
                    logger.warning(self._degraded_reason)
                    self._ollama_embeddings = None
                    self._active_backend = None
            if self.embedding_backend == "ollama":
                raise RuntimeError(
                    f"embedding_backend='ollama' was requested but is unavailable: {self._degraded_reason}"
                )

        if self.embedding_backend in ("auto", "sentence_transformers"):
            if self._init_sentence_transformer_backend():
                embeddings = self._encoder.encode(texts, normalize_embeddings=True)
                self._faiss_index = self._build_vector_index(embeddings)
                return

        # TF-IDF fallback -- no network, no model download, always available.
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

        if self._active_backend in ("ollama", "sentence_transformers"):
            if self._active_backend == "ollama":
                raw = np.asarray([self._ollama_embeddings.embed_query(query)], dtype=np.float32)
            else:
                raw = np.asarray(self._encoder.encode([query], normalize_embeddings=True), dtype=np.float32)
            raw = raw / np.clip(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12, None)
            scores, indices = self._faiss_index.search(raw, min(k, len(self.documents)))
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
