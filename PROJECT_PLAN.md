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

- [x] `src/data/emotion_stress_map.py` bridge finished: projects face/speech emotion logits
      onto the shared Healthy→Severe axis via the two (distinct) mapping tables (done in P0;
      consumed by `FusionPairDataset`'s matched-emotion sampling)
- [x] `src/models/fusion.py` — gated cross-modal attention Transformer over the three 256-d
      embeddings, trained with modality dropout (graceful missing-modality degradation),
      exposes per-sample modality contribution weights (the "72% physio / 20% face / 8%
      voice" meter). Verified: shapes, backward pass, explicit missing-modality masking.
- [x] `src/models/heads.py` — shared fused representation → 4-class softmax head (MC-dropout
      uncertainty via `predict_with_uncertainty`) + 3-output regression head (range-scaled
      to the documented 0-34/0-24/0-39 bounds), plus the score↔class **consistency term**
      (src/train/losses.py). Verified: MC-dropout variance scales with dropout rate.
- [x] Joint multi-task loss: class-balanced focal (classification) + Huber (regression),
      auto-balanced via `UncertaintyWeightedMultiTaskLoss` (learned log-variance per task)
- [x] `src/train/train_fusion.py` — anchored training on the labelled tabular rows via
      `FusionPairDataset` (weak-pairing by matched-emotion sampling, resampled every epoch),
      with a **tabular-only fallback path** (face+speech masked via the same modality-dropout
      mechanism) reported side-by-side with full fusion every validation epoch, so numbers
      never conflate the two. Frozen-encoder eval-mode bug caught and fixed during
      verification (requires_grad=False alone doesn't stop BatchNorm/Dropout drift under
      Lightning's per-epoch `.train()` call).
- [ ] First full end-to-end Macro-F1 / RMSE numbers — **pending real data**; the training
      loop itself is verified end-to-end (fast_dev_run against synthetic weakly-paired
      batches: forward, backward, both fusion and tabular-only metrics logged correctly)

**If blocked:** same data/GPU caveat as P1. Additionally, if wav2vec2 fine-tuning proves
too slow/heavy in the eventual training environment, fall back to the CNN-BiLSTM speech
encoder (already built as the P1 fallback) — fusion code is encoder-agnostic on the 256-d
embedding contract, so this swap costs nothing downstream.

---

## P3 — Explainability + trust layer (Objective 3 — not optional)

- [x] `src/explain/gradcam.py` — Grad-CAM (hooks `FaceEmotionEncoder.gradcam_target_layer`)
      + Score-CAM, implemented from scratch on torch hooks (no extra dependency). Verified:
      correct heatmap shape/range, differs by target class, doesn't leak train/eval mode.
- [x] `src/explain/ig_audio.py` — Integrated Gradients (Captum) over the speech waveform,
      with a documented vanilla-gradient-saliency fallback if Captum is unavailable, plus
      `pool_attribution_to_frames` for the "which time-frequency regions mattered" summary.
      Verified against the CNN-BiLSTM encoder.
- [x] `src/explain/shap_tab.py` — SHAP (TreeExplainer for LightGBM, GradientExplainer for
      the FT-Transformer/residual-MLP torch encoders) + `combine_shap_and_attention` merging
      SHAP with the FT-Transformer's own attention weights. Verified against both backbones.
- [x] Fusion attention exposed as the modality-contribution meter (face % / speech % /
      physio %) — `GatedCrossModalFusion.forward`'s `modality_weights` output (P2)
- [x] `src/explain/masked_distress.py` — the **Masked-Distress Index**:
      `face_calm * max(voice_high_arousal, physio_high_arousal)`, documented as a
      CortexAI-original, clinically-unvalidated heuristic (flagged for real-data
      validation, consistent with the P4 placeholder-KB caveat). Verified: fires on
      face-calm/voice-or-physio-high-arousal, stays low when face already reads distressed
      or everything is genuinely calm.
- [x] `src/explain/counterfactual.py` — DiCE counterfactuals ("if sleep quality rose 2→4,
      prediction drops to Mild") using DiCE's "random" search method (its default
      "gradient" method fails against BatchNorm1d layers with a 1-D-input error — verified
      and documented in code, not silently swapped), plus a dependency-free single-feature
      grid-search fallback verified against a known decision boundary.
- [x] `src/explain/conformal.py` — split-conformal **Adaptive Prediction Sets** (primary,
      dependency-free; empirically verified >=90% coverage at alpha=0.1, and that
      uncertain predictions yield strictly larger sets than confident ones — the naive LAC
      threshold method was tried first and rejected because it collapsed both cases to
      identical singleton sets) + a MAPIE `SplitConformalClassifier` alternate path (MAPIE's
      1.x API differs completely from the pre-1.0 `MapieClassifier` the docs' tech-stack
      list implies; pinned `mapie>=1.0.0` and documented the new contract). `UncertaintyGate`
      combines this with `heads.py`'s MC-dropout confidence to decide human deferral.

**If blocked:** every method here has a documented degraded mode if the primary library is
unavailable in the target training environment (e.g. Captum IG → simple gradient
saliency; DiCE → grid-search counterfactual as fallback) — noted inline in each module,
not silently swapped. Two real issues were hit and fixed during verification rather than
guessed around: DiCE's gradient method incompatibility with BatchNorm, and MAPIE's 0.x→1.x
breaking API change.

---

## P4 — RAG clinical layer + agent orchestrator + safety

- [x] `data/knowledge_base/` — curated/stubbed clinical reference text (DSM-5 criteria
      excerpts, DASS-21 interpretation bands, WHO guidance, crisis resources). **Not in
      `./docs`, so this is sourced/stubbed and clearly marked as placeholder content
      requiring a real clinical KB review before any real-world use** — see
      `data/knowledge_base/README.md` for the full per-file disclosure. 19 entries.
- [x] `src/reasoning/retriever.py` — FAISS + sentence-transformers embeddings (primary) with
      automatic TF-IDF fallback (scikit-learn, no model download) when the sentence-
      transformers model can't be fetched — verified: HF Hub is unreachable from this
      sandbox (same restriction as wav2vec2), so the fallback path is what's actually
      exercised and tested here; the primary path activates wherever HF Hub access exists.
- [x] `src/reasoning/rag_report.py` — cited, grounded narrative report generation (Claude
      API, injectable client for testing). Every clinical claim traces to a retrieved
      source; `validate_citations` mechanically rejects fabricated `[doc-id]` markers and
      falls back to the always-valid cached report rather than surfacing an unsourced claim
      — verified with a fake LLM client returning both valid and hallucinated citations.
- [x] `src/reasoning/agent_graph.py` — thin LangGraph orchestrator: preprocess → predict →
      uncertainty-gate → explain → retrieve → report, plus `answer_follow_up` for
      interactive questions ("why moderate?") that reuses the existing prediction rather
      than re-predicting. Coordinates only — never overrides the model's call. Verified
      end-to-end against the real knowledge base with fake model-stage functions.
- [x] Crisis/helpline resource surfacing wired in wherever severe-distress indicators
      co-occur (predicted-Severe class OR high Masked-Distress Index) — the agent graph's
      `retrieve` node force-attaches crisis-resource documents via direct category lookup
      rather than leaving it to similarity-search ranking chance; verified both trigger
      paths independently. Decision-support (not diagnosis) framing is baked into both the
      live-LLM system prompt and the cached-report template — checked again in P6.

**If blocked:** no ANTHROPIC_API_KEY in this sandbox → the live-LLM path (prompt
construction, citation validation, hallucination fallback) is verified with an injected
fake client; the cached-report fallback (docs/MINDSCOPE_Blueprint.pdf's documented "live
RAG → pre-generated cached reports" fallback) is what's actually exercised end-to-end and
is what the agent graph produces by default in this environment, always clearly labeled
`cached=True` rather than silently presented as a live report.

---

## P5 — Frontend + API integration

- [x] Extract `DESIGN.md` tokens into `frontend/tailwind.config.ts` exactly (colors,
      Inter + JetBrains Mono, 8px spacing scale, radii, elevation) — spot-checked hex
      values against DESIGN.md directly, match exactly
- [x] Scaffold `frontend/` (Vite + React 18 + TS + Tailwind), Material Symbols + the two
      Google Fonts loaded the same way the source `code.html` files load them
- [x] Convert each of the six `docs/frontend_design/*/code.html` pages into a route/page
      component, 1:1 on layout/color/spacing/copy — **no redesign**: Dashboard,
      New AI Assessment, Explainable AI Insights (expanded variant implemented as an
      in-page `expanded` state, not a separate route, after diffing the two source HTML
      files and finding the expanded version only appends 3 sections + swaps one column),
      Clinical Report, Population Analytics
- [x] Extract repeated header/sidebar markup into shared layout components
      (`frontend/src/components/layout/`)
- [x] Replace every "MindScope" occurrence with "CortexAI" (branding, page titles, copy) —
      verified via grep: zero user-visible leftovers, only one code comment referencing the
      source reference-file's name
- [x] Wire real charts (Recharts) in place of static mockup content; merged the
      responsible-AI copy pattern from `MindScope_UI_Template.jsx` (the six approved Stitch
      pages carried none themselves, and P4/P6 require it non-negotiably) — sidebar
      disclaimer, Clinical Report footer, and a `CrisisResourcesCard` surfaced on
      severe-flagged results, flagged as the one place content was added beyond a literal
      1:1 conversion
- [x] `src/api/` — FastAPI (`main.py`, `schemas.py`, `inference.py`): `/assess`,
      `/explain/{id}`, `/counterfactual/{id}`, `/report/{id}`, `/follow-up`, `/health`;
      every prediction-carrying response includes the decision-support disclaimer and
      uncertainty-gate result, and flags `is_demo_untrained_model` when (as in this sandbox)
      no trained checkpoint is loaded, rather than presenting untrained-model output as real
- [ ] Wire frontend → real API (currently the frontend's typed mock-data layer,
      `frontend/src/lib/mockData.ts`, stands in — shaped to match the API's actual response
      schemas so swapping is a data-source change, not a restructuring) — next step
- [ ] `src/eval/fairness_audit.py` — subgroup metrics by RAVDESS actor gender metadata

**If blocked:** if React wiring runs long, Streamlit is the documented fallback UI —
not needed; React conversion completed and verified (`npm run build` clean, dev server
boots and all 5 routes render with zero runtime errors, per the conversion agent's report,
independently re-verified: build re-run here, dev server curled directly).

---

## P6 — Evaluate, harden, polish

- [ ] Run the full metric suite from `Metrics_Used.docx` exactly, per-target for regression
      — **blocked on real data**; `src/eval/metrics.py` implements and unit-tests the exact
      suite (headline Macro-F1 / RMSE), ready to run the moment training completes
- [x] `src/eval/ablation.py` — fusion vs. each single modality, same split (empirical
      justification for the whole architecture). `fusion_beats_every_modality()` is the
      mechanical check for the headline claim; verified true/false/no-fusion-entry cases
      against synthetic metrics (real numbers pending real training runs)
- [x] `src/eval/fairness_audit.py` — per-gender (RAVDESS actor metadata) subgroup metrics
      + macro-F1/accuracy gap, reusing the exact Metrics_Used.docx suite per group; verified
      it detects both a zero gap and a real, sizeable gap on synthetic data
- [x] Responsible-AI copy audit: decision-support framing, human-review deferral copy,
      crisis-resource surfacing — grepped for all three across `src/api`, `src/reasoning`,
      `src/explain`, and `frontend/src`; all three present in the API layer, the RAG/agent
      layer, and the frontend (sidebar disclaimer, Clinical Report footer,
      `CrisisResourcesCard`)
- [x] `README.md` — setup, architecture summary, how to run, how to reproduce metrics,
      current status (what's verified vs. pending real data)
- [ ] Pitch notes — not written; the "one sentence for the judges" framing and the 3-minute
      demo choreography are already specified in `docs/MINDSCOPE_Blueprint.pdf` Sec. 07/12
      and haven't needed adaptation

**If blocked:** full training wasn't possible in this sandbox (no GPU, no dataset files —
logged in P0 and unresolved throughout). Every phase's code is complete, real, and verified
against synthetic fixtures rather than left as stubs; every place a real metric number
would go instead says so explicitly (`is_demo_untrained_model`, `PROJECT_PLAN.md` status
notes) rather than a fabricated number being presented as a result.
