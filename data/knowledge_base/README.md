# Clinical Knowledge Base — PLACEHOLDER CONTENT

**This directory's content is placeholder/stub reference text, not a licensed
or clinically-reviewed knowledge base.** It exists so the RAG layer
(`src/reasoning/retriever.py`, `src/reasoning/rag_report.py`) has something
concrete to retrieve from and cite during development and testing.

This content was not supplied in `./docs` (the task brief for this project
says so explicitly), so it was authored/paraphrased here to unblock the RAG
pipeline. Specifically:

- **`dsm5_criteria.json`** — short, paraphrased summaries of general
  symptom *domains* associated with stress/anxiety/depressive presentations
  (sleep disruption, anhedonia, autonomic arousal, etc.), **not** verbatim
  DSM-5 diagnostic criteria text (which is copyrighted and was not available
  to reproduce here). Do not treat these entries as diagnostic criteria.
- **`dass21_bands.json`** — severity-band *labels* (Normal/Mild/Moderate/
  Severe/Extremely Severe) presented as illustrative interpretation
  language, using this project's own score ranges (Depression 0–34, Anxiety
  0–24, Stress 0–39, per `docs/1.docx`) divided into proportional bands.
  These band cut-points are **not** the published DASS-21/DASS-42 norm
  tables (those apply to the standard 0–42 doubled-score scale, which does
  not match this dataset's 0–34/0–24/0–39 ranges) — they are placeholders
  showing the *mechanism* (score → band → cited interpretation), pending a
  real psychometric calibration against this dataset.
- **`who_guidance.json`** — general, paraphrased public-health framing
  (screening is not diagnosis, encourage professional follow-up, stigma
  reduction) in the spirit of WHO mental-health communications, not a
  verbatim WHO publication excerpt.
- **`crisis_resources.json`** — a small set of well-known crisis lines
  (e.g. the US 988 Suicide & Crisis Lifeline) plus generic
  "contact local emergency services" guidance. This is **not** an
  exhaustive or region-verified directory — a real deployment must replace
  this with a properly maintained, geo-aware crisis-resource service.

**Before any real-world use:** replace every file in this directory with
licensed/verified source text (actual DSM-5 excerpts under appropriate
license, the published DASS-21 interpretation tables mapped correctly to
whatever scoring convention the real deployment uses, an official WHO
source document, and a maintained regional crisis-resource directory), and
have it reviewed by a qualified clinician. The RAG layer is built so that
swapping this content requires no code changes — only re-running the
indexer (`src/reasoning/retriever.py`).
