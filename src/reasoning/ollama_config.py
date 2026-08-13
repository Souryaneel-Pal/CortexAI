"""Shared configuration and health-checking for the local Ollama stack.

The RAG layer runs entirely on a local Ollama instance (no data leaves the
machine, which matters for a system handling face/voice/physiological input):

  - LLM:        `llama3.1`          -> cited clinical narratives, follow-up answers
  - Embeddings: `nomic-embed-text`  -> the FAISS index over the clinical KB

Everything here exists to make "Ollama isn't running" a *clean, fast, and
visible* degradation rather than a hang or a 500. Two failure modes are
handled distinctly, because they need different messages:

  - **unreachable** -- nothing listening on the Ollama port. The whole
    reasoning stack degrades: retrieval falls back to TF-IDF, reports fall
    back to the templated cached generator.
  - **model_missing** -- Ollama is up but the requested model was never
    pulled. This is by far the most common setup mistake, and the fix is one
    command, so the message says exactly which command.

`probe_ollama()` is deliberately cheap (a GET to /api/tags with a short
timeout) and its result is cached for `_PROBE_TTL_SECONDS`, so a request
arriving while Ollama is down costs milliseconds instead of a socket
timeout per node in the agent graph.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "llama3.1"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

# How long a cached probe result stays fresh. Short enough that starting
# Ollama mid-session is picked up without a restart.
_PROBE_TTL_SECONDS = 30.0
_PROBE_TIMEOUT_SECONDS = 3.0
# Generation is the slow call; a local 8B model on CPU can legitimately take
# tens of seconds, but a request must not hang a web worker indefinitely.
DEFAULT_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("CORTEXAI_OLLAMA_TIMEOUT", "120"))


def base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def llm_model() -> str:
    return os.environ.get("CORTEXAI_OLLAMA_LLM_MODEL", DEFAULT_LLM_MODEL)


def embedding_model() -> str:
    return os.environ.get("CORTEXAI_OLLAMA_EMBED_MODEL", DEFAULT_EMBEDDING_MODEL)


@dataclass
class OllamaStatus:
    """Result of a health probe against the local Ollama instance."""

    reachable: bool
    base_url: str
    models: list[str] = field(default_factory=list)
    error: str | None = None

    def has_model(self, name: str) -> bool:
        """Ollama reports models tagged (`llama3.1:latest`); a bare name
        should match its `:latest` tag, and vice versa.
        """
        if not self.reachable:
            return False
        wanted = name if ":" in name else f"{name}:latest"
        return any(m == name or m == wanted or m.split(":")[0] == name.split(":")[0] for m in self.models)

    def unavailable_reason(self, model: str) -> str | None:
        """A user-facing explanation, or None when `model` is ready to use."""
        if not self.reachable:
            return (
                f"Cannot reach Ollama at {self.base_url} ({self.error}). "
                f"Start it with `ollama serve`, then retry."
            )
        if not self.has_model(model):
            available = ", ".join(sorted(self.models)) or "none"
            return (
                f"Ollama is running at {self.base_url} but the model '{model}' is not installed. "
                f"Run `ollama pull {model}`. Models currently available: {available}."
            )
        return None


_probe_cache: tuple[float, OllamaStatus] | None = None
_probe_lock = threading.Lock()


def probe_ollama(force: bool = False) -> OllamaStatus:
    """Cheap, cached health probe. Never raises -- an unreachable Ollama is a
    normal operating condition here, not an exception.
    """
    global _probe_cache

    with _probe_lock:
        if not force and _probe_cache is not None:
            probed_at, cached = _probe_cache
            if time.monotonic() - probed_at < _PROBE_TTL_SECONDS:
                return cached

    url = base_url()
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=_PROBE_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read())
        models = [m["name"] for m in payload.get("models", [])]
        status = OllamaStatus(reachable=True, base_url=url, models=models)
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        status = OllamaStatus(reachable=False, base_url=url, error=f"{type(exc).__name__}: {exc}")

    with _probe_lock:
        _probe_cache = (time.monotonic(), status)
    return status


def reset_probe_cache() -> None:
    """Drop the cached probe result (used by tests, and after a config change)."""
    global _probe_cache
    with _probe_lock:
        _probe_cache = None
