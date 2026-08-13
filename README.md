# 🧠 CortexAI - Explainable Multimodal Psychiatric AI

> A local-first, explainable multimodal deep-learning framework that reads face, voice, and body signals together to estimate a person's mental-health state. It classifies stress severity, estimates Depression, Anxiety, and Stress scores, and explains every prediction as grounded, clinician-ready evidence.

> **⚠️ IMPORTANT NOTICE:** This system is a research prototype and decision-support tool designed to assist qualified professionals, **not a standalone diagnostic device**. Every surface states its limitations, and low-confidence predictions are deferred to human review.

---

## Table of Contents

1. [Demo Recordings](#demo-recordings)
2. [Architecture & Pipeline](#architecture--pipeline)
3. [Key Innovations & Differentiators](#key-innovations--differentiators)
4. [Evaluation Metrics & Realities](#evaluation-metrics--realities)
5. [Application Flow & Sign-In](#application-flow--sign-in)
6. [Admin Model Settings](#admin-model-settings)
7. [Local Reasoning Stack & Report Provenance](#local-reasoning-stack--report-provenance)
8. [No Fabricated Data](#no-fabricated-data)
9. [Tech Stack](#tech-stack)
10. [API Reference](#api-reference)
11. [Quick Start & Setup](#quick-start--setup)
12. [Security Posture](#security-posture)
13. [Project Structure](#project-structure)

---

## Demo Recordings

> GitHub renders the players below inline. If your viewer strips the `<video>`
> tag (some mirrors and offline Markdown readers do), use the download links.

### 1 · End-to-end assessment
Upload or capture a face, add a voice clip, enter the clinical indicators, and run the multimodal assessment.

<video src="https://github.com/SouryaneelPal/CortexAI/raw/main/docs/media/01-assessment-walkthrough.mp4" controls muted playsinline width="100%"></video>

[⬇ Download 01-assessment-walkthrough.mp4](docs/media/01-assessment-walkthrough.mp4)

### 2 · Explainability and the grounded report
Grad-CAM over the submitted face, signed SHAP across the 18 features, the Masked-Distress Index, and the Ollama-generated cited narrative.

https://github.com/SouryaneelPal/CortexAI/raw/main/docs/media/02-explainability-and-report.mp4

[⬇ Download 02-explainability-and-report.mp4](docs/media/02-explainability-and-report.mp4)

### 3 · Integrated camera and admin settings
Live camera capture, plus the Admin-only model controls (uncertainty gate, MDI sensitivity, modality overrides).

<video src="https://github.com/SouryaneelPal/CortexAI/raw/main/docs/media/03-camera-and-settings.mp4" controls muted playsinline width="100%"></video>

[⬇ Download 03-camera-and-settings.mp4](docs/media/03-camera-and-settings.mp4)

---

## Architecture & Pipeline

CortexAI operates on a two-stage fusion architecture that respects the fact that the provided datasets are not natively row-paired.

```text
+---------------------------------------------------------------------------------+
|                              MULTIMODAL INPUTS                                  |
|   [ Facial Image: 48x48 ]   [ Speech: .wav ]   [ Tabular: 18 Features ]         |
+-----------+--------------------------+-------------------------+----------------+
            |                          |                         |
+-----------v-----------+  +-----------v----------+  +-----------v----------+
|    VISION ENCODER     |  |    AUDIO ENCODER     |  |   TABULAR ENCODER    |
|  EfficientNet-B0+CBAM |  |    Wav2Vec2-base     |  |    FT-Transformer    |
+-----------+-----------+  +-----------+----------+  +-----------+----------+
            |                          |                         |
+-----------v--------------------------v----------+              |
|            EMOTION-TO-STRESS BRIDGE             |              |
|  (Projects 7-way FER & 8-way RAVDESS to 4-tier) |              |
+--------------------------+----------------------+              |
                           |                                     |
+--------------------------v-------------------------------------v----------------+
|                  GATED CROSS-MODAL ATTENTION FUSION                             |
|      (Learns modality reliability weights, handles missing modalities)          |
+--------------------------+-------------------------------------+----------------+
                           |                                     |
+--------------------------v----------+            +-------------v--------+
|        CLASSIFICATION HEAD          |            |   REGRESSION HEAD    |
|     4-Class (Healthy -> Severe)     |            | 3 Scores (Dep/Anx/Str)|
+--------------------------+----------+            +-------------+--------+
                           |                                     |
+--------------------------v-------------------------------------v----------------+
|                     TRUST & EXPLAINABILITY STACK                                |
|  [ Grad-CAM ]  [ Integrated Gradients ]  [ SHAP ]  [ Masked-Distress Index ]    |
|  [ Uncertainty Gate (MC-Dropout, operator-configurable threshold) ]             |
+------------------------------------------+--------------------------------------+
                                           |
+------------------------------------------v--------------------------------------+
|                    RAG CLINICAL NARRATIVE & AGENT                               |
|   (LangGraph Orchestrator + Vector Index + Local Ollama Llama-3.1)              |
+---------------------------------------------------------------------------------+
```

---

## Key Innovations & Differentiators

* **Masked-Distress Index (MDI):** A novel cross-modal contradiction score. When the face reads calm but voice/physiology indicate high arousal, MDI quantifies the gap to flag suppressed distress. *(This formula is CortexAI's own construction and is clinically unvalidated.)*
* **Uncertainty "I'm not sure" Gate:** MC-dropout confidence against an operator-configurable threshold. The system defers to a human clinician when confidence is low.
* **Missing-Modality Resilience:** Trained with modality dropout, the cross-modal attention fusion degrades gracefully if a webcam or mic is missing — the same mechanism powers the admin "ignore this modality" overrides.
* **Zero-Hallucination Reports:** RAG over a local clinical knowledge base (DASS-21/DSM-5). A narrative is **rejected and replaced** — not patched — if it cites a source that wasn't retrieved, cites *nothing at all*, or drops the decision-support framing.
* **100% Local Execution (Privacy First):** Biometric data (face, voice, physiology) and LLM inference (via Ollama `llama3.1`) run entirely on-device.
* **Provenance by construction:** Every report declares which generator wrote it, and every fabricated shortcut has been removed from the data layer (see [No Fabricated Data](#no-fabricated-data)).

---

## Evaluation Metrics & Realities

All six phases are code-complete, with modality encoders and the fusion stack trained directly on the Apple-silicon MPS backend.

### Unimodal Performance (Held-out Validation Splits)

| Modality Encoder | Task / Target | Val Split | Accuracy | Macro-F1 | RMSE |
| --- | --- | --- | --- | --- | --- |
| **Face** (EfficientNet-B0) | 7-way FER emotion | Stratified 15% | 0.615 | **0.604** | - |
| **Speech** (Wav2Vec2) | 8-way RAVDESS emotion | 4 held-out actors | 0.708 | **0.698** | - |
| **Tabular** (FT-Transformer) | 4-way Stress Class | Stratified 15% | 0.283 | **0.228** | 10.53 |

*Fairness Audit:* Speech performance across held-out RAVDESS actors yields a female Macro-F1 of **0.714** vs. a male Macro-F1 of **0.680** (gap: 0.034).

> Speech **must** be split by actor, not by clip: 24 actors speak the same two sentences, so a random clip-level split measures speaker memorisation rather than emotion recognition.

### Multimodal Proof-of-Fusion Ablation

| Source Pipeline | Macro-F1 | Weighted-F1 | RMSE (Mean) |
| --- | --- | --- | --- |
| Face Only | 0.251 | 0.326 | 11.015 |
| Speech Only | 0.251 | 0.294 | 11.900 |
| Tabular Only | 0.118 | 0.147 | **9.433** |
| **Fusion** | **0.257** | 0.301 | 11.114 |

`fusion_beats_every_modality()` returns a **computed boolean**, not an assertion — and it currently returns **False**. Fusion has the best Macro-F1 but a *worse* RMSE than tabular-only, so the "fusion wins" claim is not supported here and the code reports that rather than asserting it.

> **Crucial Data Finding:** While the face and speech emotion signals are highly learnable, the 18 provided tabular features carry effectively **no predictive signal** for the targets on this specific dataset. The largest absolute correlation between any feature and any score is **0.046**. Gradient-boosted trees reach a Macro-F1 of only 0.219 (a stratified random guess scores 0.270), and R² is negative (≈ −0.02) for all three scores. Because the tabular rows act as the ground-truth anchor, the fusion model mathematically cannot synthesize strong predictions from noisy anchors. **Treat every stress-class and severity-score number this system produces as not clinically meaningful on this dataset.** The metrics above are honest reflections of the data limits, proving that our *trust, explainability, and gating layers* operate correctly even when the predictive task is fundamentally constrained.

Two caveats on the fusion numbers:

* **Validation pairing is label-independent.** Matched-emotion weak pairing keys the sampled face/voice on the row's ground-truth label, so media paired that way encodes the answer. That is a defensible *training* prior but would make a validation score meaningless, so `FusionPairDataset(pair_by_label=False)` is used for validation.
* **The learned gate collapsed onto face** (0.93 / 0.06 / 0.01) — the fingerprint of that training-time pairing. Read the modality meter as "what the gate learned on this data", not as a clinical claim about which channel matters.

Reproduce end-to-end:

```bash
python -m src.data.validate_datasets
python -m src.train.train_modality --config configs/tabular.yaml
python -m src.train.train_modality --config configs/face.yaml
python -m src.train.train_modality --config configs/speech.yaml
python -m src.train.train_fusion   --config configs/fusion.yaml
python -m src.eval.run_evaluation      # metric suite + ablation + fairness
```

---

## Application Flow & Sign-In

```
  /  SignIn
  │
  ├─→ /dashboard          Cohort stats from real stored history + latest live result
  ├─→ /assessment/new     Upload face + audio, enter clinical metrics → POST /predict
  │        │
  │        └─→ /results   4-class prediction, DASS scores, Grad-CAM, SHAP, MDI,
  │                       audio Integrated Gradients, modality attribution
  │                       └─ [Generate Clinical Report] → runs Ollama RAG → /reports
  │
  ├─→ /reports            The cited narrative + its sources and provenance
  ├─→ /analytics          Cohort analytics from real stored history
  └─→ /settings           Admin only: uncertainty gate, MDI sensitivity, modality overrides
```

Routes other than `/` are wrapped in `ProtectedRoute` and redirect to sign-in without a token. The token is a signed JWT held in `sessionStorage` and sent as `Authorization: Bearer …`.

### Demo accounts

There is no user store — these are hardcoded demo credentials, documented here because they are not a secret:

| Role | User ID | Password | Unlocks |
| --- | --- | --- | --- |
| **Admin** | `admin` | `admin` | Everything, including **Settings** (model tuning) |
| Clinician | `julian.vance@cortex.ai` | `password` | Everything except writing Settings |

Both roles can *read* settings; only Admin can change them.

---

## Admin Model Settings

`/settings` (Admin only) writes to the `settings` table, and inference reads it **per request** — so a change takes effect on the next assessment with no restart.

| Control | Effect | Read at |
| --- | --- | --- |
| Uncertainty gate threshold | Defer to a human when confidence falls below it | `inference.py:_uncertainty` |
| MDI sensitivity | Threshold at which cross-modal contradiction is flagged, and at which crisis resources are force-attached | `inference.py:_explain`, report path |
| Ignore face / speech / tabular | Masks that modality through the fusion gate's `modality_mask` | `inference.py:_preprocess` |

The modality overrides reuse the same `modality_mask` the model was *trained* to degrade into via modality dropout, so ignoring a channel is a supported inference mode rather than a hack.

---

## Local Reasoning Stack & Report Provenance

`llama3.1` writes the cited narratives; `nomic-embed-text` embeds the clinical KB. Configure with `OLLAMA_BASE_URL`, `CORTEXAI_OLLAMA_LLM_MODEL`, `CORTEXAI_OLLAMA_EMBED_MODEL`.

`GET /health` reports this stack live, so the UI warns **before** an assessment that the narrative will be templated rather than surprising the clinician afterwards. **Nothing hard-fails when Ollama is down**, and every degradation is labelled:

| Situation | Retrieval | Report |
| --- | --- | --- |
| Ollama up, models pulled | `nomic-embed-text` + vector index | `ollama:llama3.1`, cited |
| Ollama up, model missing | falls back | template, reason names `ollama pull <model>` |
| Ollama unreachable | sentence-transformers → TF-IDF | template, reason names `ollama serve` |
| Model output rejected | — | template, reason names what was rejected |

Three orthogonal response fields, because conflating them mislabels real output:

* `generator` — `ollama:llama3.1`, `anthropic:<model>`, or `template`.
* `cached` — **this is a template, not model-written.** Drives the UI's "not model-generated" warning.
* `from_store` — served from SQLite rather than generated this request. A stored `llama3.1` narrative is `from_store: true` with `cached: false`, because it is still model-written.

> **Vector index note.** `faiss.IndexFlatIP` is used where FAISS can load safely. In the API process it cannot: faiss-cpu and PyTorch each link their own OpenMP runtime, and loading both aborts the interpreter (`OMP: Error #15`) in *either* import order — an abort, not a catchable exception. There an exact NumPy inner-product index runs instead. `IndexFlatIP` *is* brute-force exact search, so this is a numerically identical substitute, pinned by a test comparing against real FAISS in a subprocess.

---

## No Fabricated Data

The dashboard and analytics pages render **only** values computed from stored assessments. Where there is no data, the UI says so.

An earlier build seeded the database with 15 invented "historical" assessments — fake patient IDs, fake 2023 dates, invented SHAP/MDI values, and hand-written narratives stored with `report_generator="ollama:llama3.1"`. That was strictly worse than frontend mock data: mock data sat behind a `SampleDataBadge` and was visibly not real, whereas the same fabrications *inside the assessments table* are indistinguishable from genuine output and were served through `/api/dashboard` as real history.

Also removed from the analytics endpoint:

* `"AI Efficacy Score": "94%"` — invented, not computable (a live assessment has no ground truth), and contradicted by the project's own measured Macro-F1 of 0.228. Replaced by a real human-deferral rate.
* A static correlation matrix claiming Stress↔HRV = 0.89, against a measured maximum |r| of 0.046. Now computed with a real Pearson correlation.
* Emotion labels ("Agitation", "Apathy") outside any encoder's label space. Now aggregated from stored FER/RAVDESS distributions.
* A hardcoded `"+12% this month"` trend string. Now computed month-over-month.

If you have an older database:

```bash
python -m src.api.database --purge-seed          # dry run: lists what it matched
python -m src.api.database --purge-seed --apply  # delete them
```

Genuine assessments are untouched — the match requires both a known seed patient ID and a 2023 timestamp.

---

## Tech Stack

| Category | Component / Library |
| --- | --- |
| **Vision Framework** | PyTorch, `timm` (EfficientNet-B0), OpenCV |
| **Audio Framework** | Hugging Face Transformers (Wav2Vec2), `librosa`, `soundfile` |
| **Tabular & ML** | FT-Transformer, LightGBM, `imbalanced-learn` (SMOTE) |
| **Explainability (XAI)** | Captum (Integrated Gradients), Grad-CAM, SHAP, DiCE |
| **RAG & Reasoning** | LangGraph, `langchain-ollama` (llama3.1, nomic-embed-text), FAISS |
| **Persistence** | SQLAlchemy + SQLite |
| **API & Serving** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | React, Vite, TypeScript, Tailwind CSS, Recharts |

---

## API Reference

Powered by FastAPI. Interactive docs at `http://127.0.0.1:8000/docs`.

| Method | Endpoint | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/health` | public | Liveness, checkpoint state, and live Ollama status |
| `POST` | `/api/auth/login` | public | Issue a JWT for a demo account |
| `GET` | `/api/settings` | any user | Read inference settings |
| `PUT` | `/api/settings` | **Admin** | Update inference settings (takes effect next assessment) |
| `GET` | `/api/dashboard` | any user | Cohort metrics derived from stored history |
| `GET` | `/api/analytics` | any user | Cohort analytics derived from stored history |
| `POST` | `/predict` (alias `/assess`) | any user | End-to-end multimodal inference; persists the run |
| `GET` | `/explain/{session_id}` | any user | SHAP, Grad-CAM overlay, audio IG, MDI |
| `GET` | `/counterfactual/{session_id}` | any user | Smallest single-feature change that flips the class |
| `GET` | `/report/{session_id}` | any user | Generate (or serve stored) cited clinical narrative |
| `POST` | `/follow-up` | any user | Grounded, cited answer to a follow-up question |

---

## Quick Start & Setup

**1. Clone & Environment**

```bash
git clone https://github.com/Souryaneel-Pal/CortexAI.git
cd CortexAI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Local Reasoning Stack (Ollama)**

```bash
# Ensure Ollama is installed (https://ollama.com)
ollama pull llama3.1
ollama pull nomic-embed-text
```

*(Copy `.env.example` to `.env` if utilizing optional hosted fallbacks.)*

**3. Run the Backend API**

```bash
uvicorn src.api.main:app --reload --port 8000
```

**4. Run the Frontend**

```bash
cd frontend
npm install
npm run dev
```

**5. Verify**

```bash
pytest                              # 181 tests
ruff check src tests                # Python lint
cd frontend && npx tsc -b           # TypeScript typecheck
cd frontend && npx oxlint .         # JS/TS lint
```

### Dataset

Distributed by the organisers and **not committed** — `data/raw/` is git-ignored. Arrange as `Extracted_images/<Emotion>/` (28,709), `Audios/Actor_01..24/` (1,440 unique), and `mental_health_multimodal.csv` (4,000 rows), then run `python -m src.data.validate_datasets`.

> The archive ships `Audios/audio_speech_actors_01-24/`, a **byte-identical duplicate** of the 24 `Actor_XX/` folders (2,880 files on disk, 1,440 unique — verified by md5). The loader de-duplicates by filename, so either layout yields 1,440 clips.

---

## Security Posture

This is a hackathon build and is **not production-ready**. Specifically:

* Credentials are hardcoded demo accounts; there is no user store, registration, or password reset.
* `CORTEXAI_JWT_SECRET` defaults to a development value. **Set it** before any real deployment — with the default, anyone holding the source can mint a valid Admin token.
* CORS is `allow_origins=["*"]`.
* Live session state is an in-process dict; assessments persist to SQLite but session state does not survive a restart or scale past one worker.
* Uploaded face/audio base64 is persisted to the local database, and `.db` files (including dated backups) are git-ignored for that reason.
* `data/knowledge_base/` is **placeholder** content and must be replaced with a licensed, clinically-reviewed KB before real-world use.

---

## Project Structure

```text
CortexAI/
├── docs/                   # Hackathon constraints, metrics schemas, design assets
├── data/                   # raw/ (git-ignored), knowledge_base/ (Clinical RAG DB), cortexai.db
├── src/
│   ├── data/               # loaders.py, emotion_stress_map.py, augment.py
│   ├── models/             # face_cnn.py, speech_net.py, tabular_ft.py, fusion.py, heads.py
│   ├── train/              # train_modality.py, train_fusion.py, losses.py
│   ├── explain/            # gradcam.py, ig_audio.py, shap_tab.py, masked_distress.py
│   ├── reasoning/          # ollama_config.py, agent_graph.py, rag_report.py, retriever.py
│   ├── eval/               # metrics.py, ablation.py, fairness_audit.py, run_evaluation.py
│   └── api/                # main.py (FastAPI + auth), inference.py, schemas.py,
│                           # database.py, dashboard_metrics.py
├── frontend/src/
│   ├── pages/              # SignIn, Dashboard, NewAssessment, Results,
│   │                       # ClinicalReport, PopulationAnalytics, Settings
│   ├── lib/                # api.ts, assessmentContext.tsx, sessionStore.ts
│   └── components/         # layout/ + ui/
├── configs/                # YAML hyperparameter configurations
├── tests/                  # Pytest suite (181 passing tests)
└── requirements.txt        # Pinned dependency graph
```
