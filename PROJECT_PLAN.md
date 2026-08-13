# CortexAI — Project Plan

Explainable multimodal deep-learning system for psychiatric/mental-health screening.
Built for Hack4Health. Source of truth for requirements: `./docs/` (read in full before
this plan was written — see "Source documents" below).

**Renamed from "MindScope" (prior planning-pass name) to "CortexAI."** Design tokens,
color system, and technical architecture from the MindScope planning pass are kept
unchanged; only the name changes, everywhere it's user-visible or in generated docs.

This plan has no hour-boxes. The Blueprint phases the work against a live 36-hour
hackathon clock with a multi-person team; this build runs as a single async engineering
session instead, so phases are ordered by dependency, not by clock time, and are worked
**phase by phase, committing incrementally per phase**.

## Known constraint, logged up front

This sandbox has **no GPU** and **no dataset files** (`data/raw/` is empty but
git-ignored and ready to receive them — see `.gitignore`). The Google Drive dataset link in `1.docx` is
blocked by this environment's network egress policy. Per explicit direction: **the user will
supply the datasets** (FER images, RAVDESS clips, the 4000-row tabular CSV) directly into
`data/raw/`. Until then:
- Every module is written to be **spec-correct and immediately runnable** against the real
  data the moment it lands — nothing is a placeholder stub.
- Tests and pipeline smoke-checks run against small **synthetic fixtures** that match the
  real schema exactly (same column names, same image/audio shapes, same filename
  convention), so the mechanics (shapes, losses, metric computation, API contracts) are
  proven correct without fabricating real training results.
- No metric numbers are ever reported as final/trained results until they come from a run
  against the real data. Anywhere a number would be shown before that (demo/UI), it is
  clearly labeled as mock/sample data, not a claimed result.

## Source documents (`./docs/`)

- `Problem_Statement.docx` — the three objectives (classification / regression / explainability)
- `1.docx` — dataset schema (FER, RAVDESS, 18-feature table) + emotion→stress mapping tables
  (facial and speech mappings differ and are kept as separate modality-specific priors)
- `Metrics_Used.docx` — exact evaluation metrics (classification + regression)
- `MINDSCOPE_Blueprint.pdf` — architecture, tech stack, repo layout, phase plan
- `Proposal.pdf` — objectives, dataset understanding, architecture (cross-checked against
  the Blueprint; consistent)
- `MindScope_UI_Template.jsx` — secondary reference for frontend feature/interaction ideas
- `frontend_design/` — six approved Stitch-exported static pages + `DESIGN.md` token spec.
  **This is the UI. We convert it to React; we do not redesign it.**

## Exact metrics (Metrics_Used.docx) — the scoring contract

- **Classification:** Accuracy, Precision, Recall/Sensitivity, F1-Score, Macro F1-Score,
  Weighted F1-Score, ROC-AUC, Confusion Matrix.
- **Regression** (per target — Depression, Anxiety, Stress): MAE, MSE, RMSE, R² Score,
  Explained Variance Score.
- **Headline metrics** (because classes are imbalanced): **Macro-F1** for classification,
  **RMSE** for regression — not raw accuracy/MSE.

---

## P0 — Frame & scaffold

- [x] Read every file in `./docs`, quote objectives + metrics back for alignment (done —
      confirmed in conversation)
- [x] Move source docs into `docs/`, unzip Stitch export into `docs/frontend_design/`
- [x] Write this file
- [x] Create full repo directory structure (`src/`, `frontend/`, `configs/`, `notebooks/`,
      `tests/`, `artifacts/`, `data/knowledge_base/`)
- [x] `src/data/emotion_stress_map.py` — both mapping tables from `1.docx`, kept separate
      (facial ≠ speech), with a documented reconciliation contract for the fusion layer
- [x] `src/data/schemas.py` — the exact 18 tabular feature names + 3 targets + 4-class label,
      typed
- [x] `src/eval/metrics.py` — the metrics harness implementing the exact list above, unit-tested
      against known-answer synthetic cases (16/16 + suite passing)
- [x] `src/data/loaders.py`, `augment.py` — FER/RAVDESS/tabular `Dataset` classes, correct
      against the documented schema/filename convention, SMOTE hook for tabular/FER imbalance
- [x] `src/data/validate_datasets.py` — script that sanity-checks `data/raw/` once populated
      (counts, class balance, corrupt-file check) — reproduces the imbalance numbers from the
      docs (FER Disgust 436, RAVDESS Neutral 96, etc.) as an assertion, not a guess
- [x] `configs/` Hydra-style YAMLs: `face.yaml`, `speech.yaml`, `tabular.yaml`, `fusion.yaml`
- [x] Synthetic fixtures generated on the fly in `tests/conftest.py` (schema-correct FER
      images, RAVDESS-named .wav clips, tabular CSV) + pytest suite proving loaders +
      metrics + emotion map + augmentation run end-to-end (25/25 passing)
- [x] `requirements.txt` reconciled against the full stack (Sec. 08 of the Blueprint) —
      added `transformers`, `langgraph`, `onnxruntime`, `python-multipart`, test tooling
- [x] `docker-compose.yml`, `Dockerfile`, `dvc.yaml` scaffold

**If blocked:** none expected — this phase is pure scaffolding and does not need real data
or a GPU.

---

## P1 — Per-modality baselines

- [x] `src/models/face_cnn.py` — EfficientNet-B0 (timm) + CBAM attention, fine-tuned head for
      7-way FER emotion, exposes penultimate 256-d embedding + Grad-CAM hook points.
      **Fallback (if blocked on timm/pretrained weight download):** 4-block CNN from scratch,
      documented as such in code and README. Both backbones verified with real forward/
      backward passes.
- [x] `src/models/speech_net.py` — Wav2Vec2-base (HF Transformers) fine-tuned for 8-way
      RAVDESS emotion, 256-d embedding.
      **Fallback (if blocked on downloading wav2vec2 weights, or too slow on CPU):**
      CNN-BiLSTM over log-Mel spectrograms + SpecAugment (torchaudio) — verified, wav2vec2
      path not downloadable in this sandbox (HF Hub not reachable) so verified by code
      inspection only, wired identically to the CNN-BiLSTM path via the same `embed_dim=256`
      contract.
- [x] `src/models/tabular_ft.py` — FT-Transformer (feature tokenizer + transformer) on the
      18 features → 256-d embedding + 4-class/3-score heads, stacked with LightGBM.
      **Fallback:** residual MLP + BatchNorm. Both verified with real forward/backward passes,
      feature-attention weights checked.
- [x] `src/train/losses.py` — class-balanced focal loss (FER/RAVDESS imbalance), Huber loss
      for scores, uncertainty-weighted multi-task loss, consistency loss (used from P2 on)
- [x] `src/train/train_modality.py` — single-modality training loop (Lightning), config-driven,
      logs the full metrics.py suite per modality (num_classes generalized: 7-way FER /
      8-way RAVDESS / 4-way stress) — **these baseline runs double as the ablation study**
      referenced in P6. All three modality training loops (face/speech/tabular) verified
      end-to-end with real DataLoader batches via `fast_dev_run`.
- [x] Class imbalance handling: class-balanced focal loss wired in for FER Disgust
      (436 vs 7215) and RAVDESS Neutral (96 vs 192); SMOTE hook for tabular (src/data/augment.py)

**If blocked:** no GPU/data in this sandbox → all three `train_modality.py` loops are
verified end-to-end (real DataLoader batches, forward/backward, loss finiteness, full
metrics-suite logging) against synthetic, schema-correct fixtures, but **not trained on
real data** until the user supplies `data/raw/`. Training is then a single
`train_modality.py --config configs/<modality>.yaml` invocation away. wav2vec2 weight
download was not reachable from this sandbox (HF Hub not on the egress allowlist) — the
CNN-BiLSTM fallback was exercised instead; wav2vec2 code is written and config-selectable
but unverified by an actual run, so try it first once real network/data access exists.

---

## P2 — Fusion + multi-task core

- [ ] `src/data/emotion_stress_map.py` bridge finished: projects face/speech emotion logits
      onto the shared Healthy→Severe axis via the two (distinct) mapping tables
- [ ] `src/models/fusion.py` — gated cross-modal attention Transformer over the three 256-d
      embeddings, trained with modality dropout (graceful missing-modality degradation)
- [ ] `src/models/heads.py` — shared fused representation → 4-class softmax head (MC-dropout
      uncertainty) + 3-output regression head, plus the score↔class **consistency term**
- [ ] Joint multi-task loss: class-balanced focal (classification) + Huber (regression),
      auto-balanced via learned/uncertainty task weights
- [ ] `src/train/train_fusion.py` — anchored training on the labelled tabular rows
      (weak-pairing via matched-emotion sampling + distribution alignment from face/speech),
      with a **tabular-only fallback path** so reported metrics stay honest; coupled
      cross-modal inference path for when a real face+voice+signal trio exists
- [ ] First full end-to-end Macro-F1 / RMSE numbers (once data lands)

**If blocked:** same data/GPU caveat as P1. Additionally, if wav2vec2 fine-tuning proves
too slow/heavy in the eventual training environment, fall back to the CNN-BiLSTM speech
encoder (already built as the P1 fallback) — fusion code is encoder-agnostic on the 256-d
embedding contract, so this swap costs nothing downstream.

---

## P3 — Explainability + trust layer (Objective 3 — not optional)

- [ ] `src/explain/gradcam.py` — Grad-CAM / Score-CAM on the face encoder's last conv block
- [ ] `src/explain/ig_audio.py` — Integrated Gradients (Captum) over the speech
      spectrogram/waveform
- [ ] `src/explain/shap_tab.py` — SHAP over the 18 tabular features + FT-Transformer attention
      weights
- [ ] Fusion attention exposed as the modality-contribution meter (face % / speech % /
      physio %)
- [ ] `src/explain/masked_distress.py` — the **Masked-Distress Index**: cross-modal
      contradiction score (face reads calm/happy, voice+physiology read high-arousal) —
      the signature feature, gets its own module as directed
- [ ] `src/explain/counterfactual.py` — DiCE counterfactuals ("if sleep quality rose 2→4,
      prediction drops to Mild")
- [ ] `src/explain/conformal.py` — MAPIE conformal prediction sets + MC-dropout, feeding the
      uncertainty gate that defers low-confidence cases to a human

**If blocked:** every method here has a documented degraded mode if the primary library is
unavailable in the target training environment (e.g. Captum IG → simple gradient
saliency; DiCE → grid-search counterfactual as fallback) — noted inline in each module,
not silently swapped.

---

## P4 — RAG clinical layer + agent orchestrator + safety

- [ ] `data/knowledge_base/` — curated/stubbed clinical reference text (DSM-5 criteria
      excerpts, DASS-21 interpretation bands, WHO guidance). **Not in `./docs`, so this is
      sourced/stubbed and clearly marked as placeholder content requiring a real clinical KB
      review before any real-world use** — per explicit instruction.
- [ ] `src/reasoning/retriever.py` — FAISS (or Chroma) index + sentence-transformers
      embeddings over the KB
- [ ] `src/reasoning/rag_report.py` — cited, grounded narrative report generation (Claude
      API). Every clinical claim traces to a retrieved source — never an unsourced claim.
- [ ] `src/reasoning/agent_graph.py` — thin orchestrator (LangGraph or simple state machine):
      preprocess → predict → uncertainty-gate → explain → retrieve → report, handles
      follow-ups ("why moderate?"). Coordinates only — never makes the clinical call.
- [ ] Crisis/helpline resource surfacing wired in wherever severe-distress indicators
      co-occur, and decision-support (not diagnosis) framing wired into every report/API
      response/UI copy string — non-negotiable, checked in P6 hardening pass too

**If blocked:** if live LLM calls aren't available/affordable during iteration, fall back
to pre-generated cached reports for the demo fixtures, clearly labeled as cached — never
silently fake a live call.

---

## P5 — Frontend + API integration

- [ ] Extract `DESIGN.md` tokens into `frontend/tailwind.config.ts` exactly (colors,
      Inter + JetBrains Mono, 8px spacing scale, radii, elevation)
- [ ] Scaffold `frontend/` (Vite + React + TS + Tailwind), Material Symbols + the two Google
      Fonts loaded as in the originals
- [ ] Convert each of the six `docs/frontend_design/*/code.html` pages into a route/page
      component, 1:1 on layout/color/spacing/copy — **no redesign**:
      Dashboard, New AI Assessment, Explainable AI Insights (+ expanded variant),
      Clinical Report, Population Analytics
- [ ] Extract repeated header/sidebar markup into shared layout components
- [ ] Replace every "MindScope" occurrence with "CortexAI" (branding, page titles, copy,
      generated docs) — design tokens/colors/architecture untouched
- [ ] Wire real charts (Recharts) in place of static mockup content; merge useful
      interaction patterns from `MindScope_UI_Template.jsx` where they fit the Stitch layout
- [ ] `src/api/` — FastAPI (`main.py`, `schemas.py`, `inference.py`): `/predict`, `/explain`,
      `/report`, session endpoints; responses carry the decision-support framing and
      uncertainty-gate result
- [ ] Wire frontend → API for real predictions/explanations/reports (falling back to the
      documented mock data only where no backend result exists yet)
- [ ] `src/eval/fairness_audit.py` — subgroup metrics by RAVDESS actor gender metadata

**If blocked:** if React wiring runs long, Streamlit is the documented fallback UI —
called out explicitly rather than silently dropping frontend scope.

---

## P6 — Evaluate, harden, polish

- [ ] Run the full metric suite from `Metrics_Used.docx` exactly, per-target for regression
- [ ] `src/eval/ablation.py` — fusion vs. each single modality, same split (empirical
      justification for the whole architecture)
- [ ] Fairness audit report (per-gender, per-class)
- [ ] Responsible-AI copy audit: decision-support framing, human-review deferral copy,
      crisis-resource surfacing — verified present in UI, API responses, and generated
      reports, not just one of the three
- [ ] `README.md` — setup, architecture summary, how to run, how to reproduce metrics
- [ ] Pitch notes

**If blocked:** if full training isn't complete by this point, this phase still runs in
full against whatever partial results exist, with gaps explicitly labeled "pending real
data" rather than omitted or faked.
